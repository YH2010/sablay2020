import config, csv, os, shutil

labels_csv = os.path.sep.join([".", config.ORIGINAL_INPUT_LABELS_FILE])

# diagnosis_labels = {}
diagnosis_labels = {
    "cataract" : "C",
    "normal fundus" : "N",
    "laser spot" : "O",
    "moderate non proliferative retinopathy" : "D",
    "branch retinal artery occlusion" : "O",
    "macular epiretinal membrane" : "O",
    "mild nonproliferative retinopathy" : "D",
    "epiretinal membrane" : "O",
    "drusen" : "O",
    "vitreous degeneration" : "O",
    "hypertensive retinopathy" : "H",
    "retinal pigmentation" : "O",
    "pathological myopia" : "M",
    "myelinated nerve fibers" : "O",
    "rhegmatogenous retinal detachment" : "O",
    "lens dust" : "X",
    "depigmentation of the retinal pigment epithelium" : "O",
    "abnormal pigment " : "O",
    "post laser photocoagulation" : "O",
    "glaucoma" : "G",
    "spotted membranous change" : "O",
    "macular hole" : "O",
    "wet age-related macular degeneration" : "A",
    "dry age-related macular degeneration" : "A",
    "epiretinal membrane over the macula" : "O",
    "central retinal artery occlusion" : "O",
    "pigment epithelium proliferation" : "O",
    "diabetic retinopathy" : "D",
    "atrophy" : "O",
    "chorioretinal atrophy" : "O",
    "white vessel" : "O",
    "retinochoroidal coloboma" : "O",
    "atrophic change" : "O",
    "retinitis pigmentosa" : "O",
    "retina fold" : "O",
    "suspected glaucoma" : "G",
    "branch retinal vein occlusion" : "O",
    "optic disc edema" : "O",
    "retinal pigment epithelium atrophy" : "O",
    "severe nonproliferative retinopathy" : "D",
    "proliferative diabetic retinopathy" : "D",
    "refractive media opacity" : "O",
    "suspected microvascular anomalies" : "O",
    "severe proliferative diabetic retinopathy" : "D",
    "central retinal vein occlusion" : "O",
    "tessellated fundus" : "O",
    "maculopathy" : "O",
    "oval yellow-white atrophy" : "O",
    "suspected retinal vascular sheathing" : "O",
    "macular coloboma" : "O",
    "vessel tortuosity" : "O",
    "hypertensive retinopathy,diabetic retinopathy" : "H",
    "idiopathic choroidal neovascularization" : "O",
    "wedge-shaped change" : "O",
    "optic nerve atrophy" : "O",
    "wedge white line change" : "O",
    "old chorioretinopathy" : "O",
    "low image quality,maculopathy" : "O",
    "low image quality" : "X",
    "punctate inner choroidopathy" : "O",
    "myopia retinopathy" : "M",
    "old choroiditis" : "O",
    "myopic maculopathy" : "M",
    "chorioretinal atrophy with pigmentation proliferation" : "O",
    "congenital choroidal coloboma" : "O",
    "optic disk epiretinal membrane" : "O",
    "optic disk photographically invisible" : "X",
    "post laser photocoagulation,diabetic retinopathy,maculopathy" : "D",
    "morning glory syndrome" : "O",
    "retinal pigment epithelial hypertrophy" : "O",
    "old branch retinal vein occlusion" : "O",
    "asteroid hyalosis" : "O",
    "retinal artery macroaneurysm" : "O",
    "suspicious diabetic retinopathy" : "D",
    "suspected diabetic retinopathy" : "D",
    "glial remnants anterior to the optic disc" : "O",
    "vascular loops" : "O",
    "diffuse chorioretinal atrophy" : "O",
    "optic discitis" : "O",
    "intraretinal hemorrhage" : "O",
    "pigmentation disorder" : "O",
    "arteriosclerosis" : "O",
    "silicone oil eye" : "O",
    "retinal vascular sheathing" : "O",
    "choroidal nevus" : "O",
    "suspected retinitis pigmentosa" : "O",
    "old central retinal vein occlusion" : "O",
    "image offset" : "X",
    "diffuse retinal atrophy" : "O",
    "fundus laser photocoagulation spots" : "O",
    "suspected abnormal color of  optic disc" : "O",
    "myopic retinopathy" : "M",
    "vitreous opacity" : "O",
    "macular pigmentation disorder" : "O",
    "suspected moderate non proliferative retinopathy" : "D",
    "suspected macular epimacular membrane" : "O",
    "peripapillary atrophy" : "O",
    "retinal detachment" : "O",
    "anterior segment image" : "X",
    "central serous chorioretinopathy" : "O",
    "suspected cataract" : "C",
    "post retinal laser surgery" : "O",
    "age-related macular degeneration" : "A",
    "intraretinal microvascular abnormality" : "O",
    "no fundus image" : "X",
}

def get_diagnosis_label(diagnosis):
    if not diagnosis in list(diagnosis_labels.keys()):
        label = str(input("Label for %s? "%(diagnosis)))
        diagnosis_labels[diagnosis] = label
    return diagnosis_labels[diagnosis]

def copy_fundus_image_to_label_folder(filename, label):
    filePath = os.path.sep.join([".", config.ORIGINAL_INPUT_TRAINING_IMAGES_PATH, filename])
    classPath = os.path.sep.join([".", config.DATASET_CLASSES_PATH, label])
    if not os.path.exists(classPath):
        os.makedirs(classPath)
    shutil.copy2(filePath, classPath)

with open(labels_csv) as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        for diagnosis in row["Left-Diagnostic Keywords"].split("_"):
            label = get_diagnosis_label(diagnosis)
            copy_fundus_image_to_label_folder(row["Left-Fundus"], label)
        for diagnosis in row["Right-Diagnostic Keywords"].split("_"):
            label = get_diagnosis_label(diagnosis)
            copy_fundus_image_to_label_folder(row["Right-Fundus"], label)