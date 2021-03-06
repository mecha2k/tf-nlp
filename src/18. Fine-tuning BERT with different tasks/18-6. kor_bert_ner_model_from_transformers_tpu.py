import pandas as pd
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from transformers import BertTokenizer
from transformers import TFBertForTokenClassification
from seqeval.metrics import f1_score, classification_report

train_ner_df = pd.read_csv("../data/ner_train_data.csv")
test_ner_df = pd.read_csv("../data/ner_test_data.csv")
train_ner_df = train_ner_df[:5000]
test_ner_df = test_ner_df[:500]
print(train_ner_df)
print(test_ner_df)

train_data_sentence = [sent.split() for sent in train_ner_df["Sentence"].values]
test_data_sentence = [sent.split() for sent in test_ner_df["Sentence"].values]
train_data_label = [tag.split() for tag in train_ner_df["Tag"].values]
test_data_label = [tag.split() for tag in test_ner_df["Tag"].values]

labels = [label.strip() for label in open("../data/ner_label.txt", "r", encoding="utf-8")]
print("개체명 태깅 정보 :", labels)

tag_to_index = {tag: index for index, tag in enumerate(labels)}
index_to_tag = {index: tag for index, tag in enumerate(labels)}

tag_size = len(tag_to_index)
print("개체명 태깅 정보의 개수 :", tag_size)


def convert_examples_to_features(
    examples,
    labels,
    max_seq_len,
    tokenizer,
    pad_token_id_for_segment=0,
    pad_token_id_for_label=-100,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

    for example, label in tqdm(zip(examples, labels), total=len(examples)):
        tokens = []
        labels_ids = []
        for one_word, label_token in zip(example, label):
            subword_tokens = tokenizer.tokenize(one_word)
            tokens.extend(subword_tokens)
            labels_ids.extend(
                [tag_to_index[label_token]] + [pad_token_id_for_label] * (len(subword_tokens) - 1)
            )

        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            labels_ids = labels_ids[: (max_seq_len - special_tokens_count)]

        tokens += [sep_token]
        labels_ids += [pad_token_id_for_label]
        tokens = [cls_token] + tokens
        labels_ids = [pad_token_id_for_label] + labels_ids

        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        padding_count = max_seq_len - len(input_id)

        input_id = input_id + ([pad_token_id] * padding_count)
        attention_mask = attention_mask + ([0] * padding_count)
        token_type_id = [pad_token_id_for_segment] * max_seq_len
        label = labels_ids + ([pad_token_id_for_label] * padding_count)

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(
            len(input_id), max_seq_len
        )
        assert (
            len(attention_mask) == max_seq_len
        ), "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_id), max_seq_len
        )
        assert len(label) == max_seq_len, "Error with labels length {} vs {}".format(
            len(label), max_seq_len
        )

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)
    data_labels = np.asarray(data_labels, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), data_labels


tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

X_train, y_train = convert_examples_to_features(
    train_data_sentence, train_data_label, max_seq_len=128, tokenizer=tokenizer
)
X_test, y_test = convert_examples_to_features(
    test_data_sentence, test_data_label, max_seq_len=128, tokenizer=tokenizer
)


# TPU 작동을 위한 코드
# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)
# with strategy.scope():

model = TFBertForTokenClassification.from_pretrained(
    "klue/bert-base", num_labels=tag_size, from_pt=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.hf_compute_loss)


class F1score(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    @staticmethod
    def sequences_to_tags(label_ids, pred_ids):
        label_list = []
        pred_list = []

        for i in range(0, len(label_ids)):
            label_tag = []
            pred_tag = []

            for label_index, pred_index in zip(label_ids[i], pred_ids[i]):
                if label_index != -100:
                    label_tag.append(index_to_tag[label_index])
                    pred_tag.append(index_to_tag[pred_index])

            label_list.append(label_tag)
            pred_list.append(pred_tag)

        return label_list, pred_list

    def on_epoch_end(self, epoch, logs):
        y_predicted = self.model.predict(self.X_test)
        y_predicted = np.argmax(y_predicted.logits, axis=2)

        label_list, pred_list = self.sequences_to_tags(self.y_test, y_predicted)

        score = f1_score(label_list, pred_list, suffix=True, zero_division="warn")
        print(f"{epoch} - f1: {score * 100:04.2f}")
        print(logs)
        print(classification_report(label_list, pred_list, suffix=True))


f1_score_report = F1score(X_test, y_test)

model.fit(X_train, y_train, epochs=3, batch_size=32, callbacks=[f1_score_report])


def convert_examples_to_features_for_prediction(
    examples, max_seq_len, tokenizer, pad_token_id_for_segment=0, pad_token_id_for_label=-100
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    input_ids, attention_masks, token_type_ids, label_masks = [], [], [], []

    for example in tqdm(examples):
        tokens = []
        label_mask = []
        for one_word in example:
            subword_tokens = tokenizer.tokenize(one_word)
            tokens.extend(subword_tokens)
            label_mask.extend([0] + [pad_token_id_for_label] * (len(subword_tokens) - 1))

        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            label_mask = label_mask[: (max_seq_len - special_tokens_count)]

        tokens += [sep_token]
        label_mask += [pad_token_id_for_label]
        tokens = [cls_token] + tokens
        label_mask = [pad_token_id_for_label] + label_mask
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        padding_count = max_seq_len - len(input_id)
        input_id = input_id + ([pad_token_id] * padding_count)
        attention_mask = attention_mask + ([0] * padding_count)
        token_type_id = [pad_token_id_for_segment] * max_seq_len
        label_mask = label_mask + ([pad_token_id_for_label] * padding_count)

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(
            len(input_id), max_seq_len
        )
        assert (
            len(attention_mask) == max_seq_len
        ), "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_id), max_seq_len
        )
        assert len(label_mask) == max_seq_len, "Error with labels length {} vs {}".format(
            len(label_mask), max_seq_len
        )

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        label_masks.append(label_mask)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)
    label_masks = np.asarray(label_masks, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), label_masks


X_pred, label_masks = convert_examples_to_features_for_prediction(
    test_data_sentence[:5], max_seq_len=128, tokenizer=tokenizer
)


def ner_prediction(examples, max_seq_len, tokenizer):
    examples = [sent.split() for sent in examples]
    X_pred, label_masks = convert_examples_to_features_for_prediction(
        examples, max_seq_len=max_seq_len, tokenizer=tokenizer
    )
    y_predicted = model.predict(X_pred)
    y_predicted = np.argmax(y_predicted.logits, axis=2)

    pred_list = []
    result_list = []

    for i in range(0, len(label_masks)):
        pred_tag = []
        for label_index, pred_index in zip(label_masks[i], y_predicted[i]):
            if label_index != -100:
                pred_tag.append(index_to_tag[pred_index])

        pred_list.append(pred_tag)

    for example, pred in zip(examples, pred_list):
        one_sample_result = []
        for one_word, label_token in zip(example, pred):
            one_sample_result.append((one_word, label_token))
        result_list.append(one_sample_result)

    return result_list


sent1 = "오리온스는 리그 최정상급 포인트가드 김동훈을 앞세우는 빠른 공수전환이 돋보이는 팀이다"
sent2 = "하이신사에 속한 섬들도 위로 솟아 있는데 타인은 살고 있어요"

test_samples = [sent1, sent2]
result_list = ner_prediction(test_samples, max_seq_len=128, tokenizer=tokenizer)
print(result_list)
