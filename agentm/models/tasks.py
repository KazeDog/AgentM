from __future__ import annotations

PROPERTY_TASKS = {
    "BBBP": "classification",
    "BACE": "classification",
    "HIV": "classification",
    "ESOL": "regression",
    "FreeSolv": "regression",
    "Lipophilicity": "regression",
}

ADMET_TASKS = {
    "ClinTox": "classification",
    "Tox21": "classification",
    "SIDER": "classification",
    "CYP1A2 inhibition": "classification",
    "CYP2C9 inhibition": "classification",
    "CYP2C19 inhibition": "classification",
    "CYP2D6 inhibition": "classification",
    "CYP3A4 inhibition": "classification",
    "hERG": "classification",
    "Ames": "classification",
    "AMES": "classification",
    "DILI": "classification",
    "BBB_Martins": "classification",
    "HIA_Hou": "classification",
    "Pgp_Broccatelli": "classification",
    "Bioavailability_Ma": "classification",
    "CYP1A2_Veith": "classification",
    "CYP2C9_Veith": "classification",
    "CYP2C19_Veith": "classification",
    "CYP2D6_Veith": "classification",
    "CYP3A4_Veith": "classification",
    "CYP2C9_Substrate_CarbonMangels": "classification",
    "CYP2D6_Substrate_CarbonMangels": "classification",
    "CYP3A4_Substrate_CarbonMangels": "classification",
    "Caco2_Wang": "regression",
    "Lipophilicity_AstraZeneca": "regression",
    "Solubility_AqSolDB": "regression",
    "PPBR_AZ": "regression",
    "VDss_Lombardo": "regression",
    "HydrationFreeEnergy_FreeSolv": "regression",
    "Half_Life_Obach": "regression",
    "Clearance_Hepatocyte_AZ": "regression",
    "Clearance_Microsome_AZ": "regression",
    "LD50_Zhu": "regression",
}


def task_mode(task_type: str, dataset_name: str) -> str:
    if task_type == "property_prediction":
        return PROPERTY_TASKS.get(dataset_name, "classification")
    if task_type == "admet_prediction":
        return ADMET_TASKS.get(dataset_name, "classification")
    if task_type == "ddi_prediction":
        return "classification"
    return "classification"
