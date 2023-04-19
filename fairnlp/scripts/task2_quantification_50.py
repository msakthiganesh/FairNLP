import pandas as pd
from tqdm.notebook import tqdm
import ast

gender_words = ["he", "she", "they", "his", "her", "their", "him", "them"]
male_words = ["he", "his", "him"]
female_words = ["she", "her"]
neutral_words = ["it", "its", "me", "their", "them", "they", "we", "you", "your"]


def helper_top_n_values(n, lst):
    sorted_lst = sorted(lst)
    top_n_values = sorted_lst[-n:]
    return top_n_values


def helper_top_n_index(top_probs, probs):
    idx = []
    for ele in top_probs:
        idx.append(probs.index(ele))
    return idx


word_imp_probas_50, word_imp_toks_50 = [], []

# start for loop here

df = pd.read_csv("allenlp_predictions.csv", index_col=None)

probs, pred_words, tokens = [], [], []
for i, row in df.iterrows():
    predictions = ast.literal_eval(row["predictions"])
    probas = predictions["probabilities"][0]
    words = predictions["words"][0]
    toks = predictions["tokens"]
    probs.append(probas)
    pred_words.append(words)
    tokens.append(toks)

df["probs"] = probs
df["pred_words"] = pred_words
df["tokens"] = tokens


gender_quants = []
top_n = 5

for i, row in df.iterrows():
    gender_dict = {}
    # Male
    for pred in row["pred_words"]:
        gender_dict["occupation"] = row["occupation(0)"]
        gender_dict["participant"] = row["other-participant(1)"]
        if pred in male_words:
            gender_dict["male_pred"] = pred
            gender_dict["male_prob"] = row["probs"][row["pred_words"].index(pred)]
            word_importances_lst = ast.literal_eval(row["word_importances"])
            top_probs = helper_top_n_values(top_n, word_importances_lst)
            top_idx = helper_top_n_index(top_probs, word_importances_lst)
            top_tokens = [row["tokens"][i] for i in top_idx]
            gender_dict["male_word_imp"] = top_probs
            gender_dict["male_tokens"] = top_tokens
            break

    # Female
    for pred in row["pred_words"]:
        if pred in female_words:
            gender_dict["female_pred"] = pred
            gender_dict["female_prob"] = row["probs"][row["pred_words"].index(pred)]
            word_importances_lst = ast.literal_eval(row["word_importances"])
            top_probs = helper_top_n_values(top_n, word_importances_lst)
            top_idx = helper_top_n_index(top_probs, word_importances_lst)
            top_tokens = [row["tokens"][i] for i in top_idx]
            gender_dict["female_word_imp"] = top_probs
            gender_dict["female_tokens"] = top_tokens
            break
    gender_quants.append(gender_dict)

df_gender_quants = pd.DataFrame.from_dict(gender_quants)
df_gender_quants.dropna(inplace=True)
df_gender_quants.reset_index(inplace=True)

word_imp_probas_50.append(male_word_imp)
