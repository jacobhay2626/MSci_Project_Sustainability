from classification import classification
descs = ["HTP(ingestion)", "HTP(Inhalation)", "XVP", "Flash Point", "Boiling Point", "Resistivity",
         "Peroxide formation", "AIT", "CGP", "CLP", "Aquatic Toxicity (mg/L)"]

preds, real, matrix, report = classification(descs, "RF", "LOO")

