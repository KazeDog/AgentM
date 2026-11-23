"""Build train-only regression fault scales; never use pilot/test predictions."""
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .common import PROJECT_ROOT, file_sha256, stable_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from scripts.audit_balanced100_scientific_performance import Canonicalizer, load_admet_source
    from agentm.models.predictors import SklearnPredictor
    root=PROJECT_ROOT
    profiles={}
    with patch('socket.socket.connect',side_effect=AssertionError('Train-scale preparation is offline')):
        for dataset in ('LD50_Zhu','Lipophilicity_AstraZeneca'):
            model=root/'outputs/tdc'/dataset/'model.pkl'
            if SklearnPredictor.load(model).mode!='regression':
                raise ValueError('Expected a verified regression model')
            source=load_admet_source(dataset,model.parent,resource_root=root,canonicalize=Canonicalizer(),cache={})
            if not source['data_hash_verified']:
                raise ValueError('Training source data hash mismatch')
            train=sorted([{'row_id':r['row_id'],'label':float(r['label'])}
                          for hits in source['index'].values() for r in hits if r['split']=='train'],
                         key=lambda r:(str(r['row_id']),r['label']))
            values=np.asarray([r['label'] for r in train],dtype=float)
            if len(values)<20 or not np.isfinite(values).all():
                raise ValueError('Insufficient or nonfinite training labels')
            quantiles=np.quantile(values,[.05,.25,.50,.75,.95],method='linear').tolist()
            iqr=quantiles[3]-quantiles[1]
            if iqr<=0:
                raise ValueError('Degenerate training scale')
            profiles[dataset]={'mode':'regression','unit':'dataset_native_target_units',
                'model_path':str(model),'model_sha256':file_sha256(model),
                'data_path':source['data_path'],'data_sha256':source['data_hash_current'],
                'reproducibility_path':source['reproducibility_path'],
                'reproducibility_sha256':file_sha256(source['reproducibility_path']),
                'training_source':source['source_id'],'split_strategy':source['split_strategy'],
                'split_seed':source['random_seed'],'training_label_count':len(train),
                'training_rows_sha256':stable_hash(train),'quantile_probabilities':[.05,.25,.5,.75,.95],
                'candidate_values':quantiles,'iqr':iqr,'minimum_separation':.5*iqr,
                'scale_fit_split':'train','test_or_pilot_values_used_for_scale':False}
    write_json(args.output,{'protocol':'admet_regression_train_quantiles',
        'rule':'uniform seeded choice among q05/q25/q50/q75/q95 at least 0.5 train IQR from baseline and gold; no candidate => not applicable',
        'pilot_informed_protocol_development':True,'profiles':profiles},overwrite=False)


if __name__=='__main__':
    main()
