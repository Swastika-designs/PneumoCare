"""Sense inventory and labelled examples for clinical abbreviation disambiguation (T12).

SYNTHETIC DATA: every sentence below was written by hand for this project to demonstrate
the method. They are not real patient records and are deliberately small; accuracy on
them shows that the method works, not how it would perform in a hospital.
"""

SENSES = {
    "MS": {
        "multiple sclerosis": "demyelinating disease central nervous system relapsing remitting lesions mri "
                              "brain spinal cord optic neuritis neurology interferon disability",
        "mitral stenosis": "heart valve narrowing mitral valve rheumatic murmur echocardiogram left atrium "
                           "diastolic cardiology valve area",
        "morphine sulfate": "opioid analgesic pain dose mg intravenous iv oral every hours narcotic "
                            "respiratory depression medication",
    },
    "RA": {
        "rheumatoid arthritis": "autoimmune joint arthritis synovitis methotrexate swelling hands stiffness "
                                "rheumatology inflammatory erosions",
        "right atrium": "heart chamber atrial cardiac echocardiogram enlarged tricuspid catheter venous "
                        "pressure dilated",
        "room air": "oxygen saturation spo2 breathing supplemental nasal cannula percent sat pulse oximetry",
    },
    "PE": {
        "pulmonary embolism": "clot lung embolus dyspnea anticoagulation heparin ct angiography d-dimer dvt "
                              "thrombosis",
        "physical examination": "exam findings inspection auscultation palpation general appearance revealed "
                                "unremarkable",
        "pleural effusion": "fluid pleural space thoracentesis blunting costophrenic angle chest x-ray "
                            "decreased breath sounds",
    },
    "CVA": {
        "cerebrovascular accident": "stroke brain hemiparesis aphasia infarct neurological deficit "
                                    "thrombolysis weakness face arm",
        "costovertebral angle": "tenderness flank kidney pyelonephritis back percussion urinary renal",
    },
    "PT": {
        "physical therapy": "rehabilitation exercise strength mobility gait therapist sessions ambulation "
                            "range motion",
        "prothrombin time": "coagulation inr warfarin bleeding seconds clotting lab anticoagulant",
    },
    "CP": {
        "chest pain": "pain chest pressure radiating exertion angina troponin ecg cardiac substernal",
        "cerebral palsy": "child motor development spasticity congenital wheelchair neurologic birth "
                          "muscle tone",
    },
}

EXAMPLES = [
    ("MS", "multiple sclerosis", "MRI of the brain showed new periventricular lesions consistent with MS relapse."),
    ("MS", "multiple sclerosis", "She was started on interferon for relapsing remitting MS after optic neuritis."),
    ("MS", "multiple sclerosis", "Neurology follow-up for MS with progressive disability and spinal cord lesions."),
    ("MS", "mitral stenosis", "Echo showed severe MS with a valve area of 0.9 cm2 and an enlarged left atrium."),
    ("MS", "mitral stenosis", "Rheumatic heart disease with MS and a diastolic murmur at the apex."),
    ("MS", "mitral stenosis", "Cardiology referral for MS before mitral valve replacement."),
    ("MS", "morphine sulfate", "Give MS 4 mg IV every 4 hours as needed for severe pain."),
    ("MS", "morphine sulfate", "Pain controlled with oral MS; monitor for respiratory depression."),
    ("MS", "morphine sulfate", "The opioid dose of MS was reduced because of sedation."),
    ("RA", "rheumatoid arthritis", "Long-standing RA with swelling and morning stiffness of both hands."),
    ("RA", "rheumatoid arthritis", "RA treated with methotrexate; rheumatology notes erosions on x-ray."),
    ("RA", "rheumatoid arthritis", "Inflammatory joint pain due to RA flare with synovitis of the wrists."),
    ("RA", "right atrium", "The catheter tip is in the RA near the tricuspid valve."),
    ("RA", "right atrium", "Echocardiogram shows a dilated RA and elevated venous pressure."),
    ("RA", "room air", "Oxygen saturation 95% on RA without supplemental oxygen."),
    ("RA", "room air", "SpO2 dropped to 88% on RA so a nasal cannula was applied."),
    ("RA", "room air", "Pulse oximetry 97 percent breathing RA at rest."),
    ("PE", "pulmonary embolism", "CT angiography confirmed a PE and heparin anticoagulation was started."),
    ("PE", "pulmonary embolism", "Sudden dyspnea after surgery; elevated D-dimer raised concern for PE."),
    ("PE", "pulmonary embolism", "History of DVT complicated by PE, now on long-term anticoagulation."),
    ("PE", "physical examination", "On PE the patient was alert, auscultation unremarkable."),
    ("PE", "physical examination", "PE revealed no abnormal findings on inspection or palpation."),
    ("PE", "physical examination", "General appearance on PE: well nourished, in no distress."),
    ("PE", "pleural effusion", "Chest x-ray shows a right PE with blunting of the costophrenic angle."),
    ("PE", "pleural effusion", "Thoracentesis drained 800 ml from the PE; decreased breath sounds improved."),
    ("CVA", "cerebrovascular accident", "Admitted with left-sided hemiparesis and aphasia due to an acute CVA."),
    ("CVA", "cerebrovascular accident", "Prior CVA with residual arm weakness; on aspirin for stroke prevention."),
    ("CVA", "cerebrovascular accident", "Thrombolysis was considered for the CVA given the facial droop."),
    ("CVA", "costovertebral angle", "Fever and right CVA tenderness suggest pyelonephritis."),
    ("CVA", "costovertebral angle", "No CVA tenderness on percussion of the flank; urinary symptoms resolved."),
    ("CVA", "costovertebral angle", "Kidney infection suspected because of CVA tenderness and back pain."),
    ("PT", "physical therapy", "Referred to PT for gait training and strength exercises after knee surgery."),
    ("PT", "physical therapy", "PT sessions three times weekly to improve mobility and ambulation."),
    ("PT", "physical therapy", "The therapist reports improved range of motion with PT."),
    ("PT", "prothrombin time", "PT was 18 seconds with an INR of 2.4 on warfarin."),
    ("PT", "prothrombin time", "Coagulation labs: PT prolonged, bleeding risk discussed."),
    ("PT", "prothrombin time", "Check PT and INR before adjusting the anticoagulant dose."),
    ("CP", "chest pain", "Presented with substernal CP radiating to the left arm on exertion."),
    ("CP", "chest pain", "CP with elevated troponin; ECG showed ST depression."),
    ("CP", "chest pain", "Pressure-like CP suggestive of angina, cardiac workup ordered."),
    ("CP", "cerebral palsy", "Child with CP and spasticity uses a wheelchair for mobility."),
    ("CP", "cerebral palsy", "Congenital CP with delayed motor development and abnormal muscle tone."),
    ("CP", "cerebral palsy", "Neurologic follow-up for CP after birth injury."),
]
