import pandas as pd
import numpy as np
import json
import ast
import itertools
from tqdm.notebook import tqdm

from allennlp.interpret.saliency_interpreters import SimpleGradient
from allennlp.predictors import Predictor

from sklearn.feature_extraction.text import TfidfVectorizer

for iteration in tqdm(range(50), total=50):

    df_template = pd.read_csv("dataset/winogender/templates.tsv", delimiter="\t")
    rows_to_drop = [i for i in range(120) if i % 2 == 1]

    df_template.drop(index=rows_to_drop, axis=0, inplace=True)
    df_template.reset_index(inplace=True, drop=True)

    sent_with_occup_participant, masked_sentences = [], []
    for iter, row in df_template.iterrows():
        sent_with_occup_participant.append(
            row["sentence"]
            .replace("$OCCUPATION", row["occupation(0)"])
            .replace("$PARTICIPANT", row["other-participant(1)"])
        )

    for sent in sent_with_occup_participant:
        masked_sentences.append(
            sent.replace("$NOM_PRONOUN", "[MASK]")
            .replace("$POSS_PRONOUN", "[MASK]")
            .replace("$ACC_PRONOUN", "[MASK]")
        )

    df_template["masked_sentences"] = masked_sentences

    predictor = Predictor.from_path("models/bert-masked-lm-2020-10-07/")

    interpreter = SimpleGradient(predictor)

    predictions, word_importances = [], []

    for i, row in tqdm(df_template.iterrows(), total=df_template.shape[0]):
        preds = predictor.predict(row["masked_sentences"])
        predictions.append(preds)

        inputs = {"sentence": row["masked_sentences"]}
        interpretation = interpreter.saliency_interpret_from_json(inputs)
        word_importances.append(interpretation["instance_1"]["grad_input_1"])

    df_template["predictions"] = predictions
    df_template["word_importances"] = word_importances

    df_template.to_csv(
        f"dataset/prediction_op/allenlp_predictions_iter{iteration}.csv", index=None
    )
