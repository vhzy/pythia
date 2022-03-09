# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch

from pythia.datasets.vqa.textvqa.dataset import TextVQADataset
from pythia.utils.text_utils import word_tokenize
from pythia.common.sample import Sample
from pythia.utils.objects_to_byte_tensor import enc_obj2bytes


class M4CTextVQADataset(TextVQADataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__(
            dataset_type, imdb_file_index, config, *args, **kwargs
        )
        self._name = "m4c_textvqa"
        self.object_clsname = [x.strip() for x in list(open('/home/ubuntu/hzy/pythia/data/objects_vocab.txt','r'))]
        self.object_clsname = ['background'] + self.object_clsname

    def preprocess_sample_info(self, sample_info):
        return sample_info  # Do nothing

    def postprocess_evalai_entry(self, entry):
        return entry  # Do nothing

    def format_for_evalai(self, report):
        answer_processor = self.answer_processor

        batch_size = len(report.question_id)
        pred_answers = report.scores.argmax(dim=-1).view(batch_size, -1)
        answer_space_size = answer_processor.get_true_vocab_size()

        predictions = []
        for idx, question_id in enumerate(report.question_id):
            # collect VQA answers
            context_tokens = report.context_tokens[idx]
            answer_words = []
            pred_source = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(
                        word_tokenize(context_tokens[answer_id])
                    )
                    pred_source.append('OCR')
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )
                    pred_source.append('VOCAB')
            # join all the answer tokens with space
            # (this should be correct for almost all cases)
            pred_answer = ' '.join(answer_words).replace(" 's", "'s")
            entry = {
                "question_id": question_id.item(),
                "image_id": report.image_id[idx],
                "answer": pred_answer,
                "pred_source": pred_source,
            }
            entry = self.postprocess_evalai_entry(entry)

            predictions.append(entry)

        return predictions

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        sample_info = self.preprocess_sample_info(sample_info)
        current_sample = Sample()

        # breaking change from VQA2Dataset: load question_id
        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = str(sample_info["image_id"])
        else:
            current_sample.image_id = sample_info["image_id"]

        if self._use_features is True:
            features = self.features_db[idx]
            if 'object_tokens' not in features['image_info_0']:
                features['image_info_0']['object_tokens'] = \
                    [self.object_clsname[x] for x in features['image_info_0']['objects']] #object_tokens是一个列表，里面是这张图片里面的物体的字符串，
            current_sample.update(features)

        current_sample = self.add_sample_details(sample_info, current_sample)
        current_sample = self.add_answer_info(sample_info, current_sample)

        # only the 'max_features' key is needed
        # pop other keys to minimize data loading overhead
        for k in list(current_sample.image_info_0):
            if k != 'max_features':
                current_sample.image_info_0.pop(k)
        for k in list(current_sample.image_info_1):
            if k != 'max_features':
                current_sample.image_info_1.pop(k)
        visual_overlap_flag = torch.zeros(150, 150)
        semantic_overlap_flag = torch.zeros(150, 150)
        visual_obj_obj_relation = self.compute_similarity_by_cosine(current_sample.image_feature_0, current_sample.image_feature_0)
        semantic_obj_obj_relation = self.compute_similarity_by_cosine(current_sample.objlabel_feature_0, current_sample.objlabel_feature_0)
        visual_ocr_ocr_relation = self.compute_similarity_by_cosine(current_sample.image_feature_1[:50,:], current_sample.image_feature_1[:50,:])
        semantic_ocr_ocr_relation = self.compute_similarity_by_cosine(current_sample.context_feature_0, current_sample.context_feature_0)
        #print("visual_obj_obj_relation:",visual_obj_obj_relation.size())
        #print("semantic_obj_obj_relation:",semantic_obj_obj_relation.size())
        #print("visual_ocr_ocr_relation:",visual_ocr_ocr_relation.size())
        #print("semantic_ocr_ocr_relation:",semantic_ocr_ocr_relation.size())
        #print(current_sample.image_feature_1.size())
        obj_ocr_relation = self.overlap(current_sample.obj_bbox_coordinates, current_sample.ocr_bbox_coordinates)
        visual_overlap_flag[:100, :100] = visual_obj_obj_relation
        visual_overlap_flag[100:, 100:] = visual_ocr_ocr_relation
        semantic_overlap_flag[:100, :100] = semantic_obj_obj_relation
        semantic_overlap_flag[100:, 100:] = semantic_ocr_ocr_relation
        visual_overlap_flag[:100, 100:] = obj_ocr_relation
        visual_overlap_flag[100:, :100] = obj_ocr_relation.transpose(1, 0)
        semantic_overlap_flag[:100, 100:] = obj_ocr_relation
        semantic_overlap_flag[100:, :100] = obj_ocr_relation.transpose(1, 0)
        current_sample.visual_overlap_flag = visual_overlap_flag
        current_sample.semantic_overlap_flag = semantic_overlap_flag
        return current_sample

    def compute_similarity_by_cosine(self, x, y):
        sim = torch.matmul(x, y.transpose(1, 0))  # M x N
        abs_x = torch.sqrt((x * x).sum(-1)).reshape(x.shape[0], 1)
        abs_y = torch.sqrt((y * y).sum(-1)).reshape(1, y.shape[0])
        abs_xy = torch.matmul(abs_x, abs_y)
        sim_out = sim / (abs_xy + 1e-5)
        return sim_out

    def overlap(self, obj_box, ocr_box):  # 100 x 4
        obj_num = obj_box.size(0)
        ocr_num = ocr_box.size(0)
        obj_box = obj_box.unsqueeze(1).repeat(1, ocr_num, 1)  # 100 x 50 x 4
        ocr_box = ocr_box.unsqueeze(0).repeat(obj_num, 1, 1)
        obj_x = obj_box[:, :, 0]
        obj_y = obj_box[:, :, 1]
        obj_x2 = obj_box[:, :, 2]
        obj_y2 = obj_box[:, :, 3]
        # [ocr_w, ocr_h, ocr_x, ocr_y] = ocr_box

        ocr_x = ocr_box[:, :, 0]
        ocr_y = ocr_box[:, :, 1]
        ocr_x2 = ocr_box[:, :, 2]  # 100 x 50
        ocr_y2 = ocr_box[:, :, 3]

        flag1 = (obj_x2 <= ocr_x).byte()
        flag2 = (ocr_x2 <= obj_x).byte()
        flag3 = (obj_y2 <= ocr_y).byte()
        flag4 = (ocr_y2 <= obj_y).byte()
        out1 = flag1 | flag2
        out2 = flag3 | flag4
        overlap_flag = torch.ones_like(out1) - (out1 | out2)
        return overlap_flag

    def add_sample_details(self, sample_info, sample):
        # 1. Load text (question words)
        # breaking change from VQA2Dataset:
        # load the entire question string, not tokenized questions, since we
        # switch to BERT tokenizer in M4C and do online tokenization
        question_str = (
            sample_info['question'] if 'question' in sample_info
            else sample_info['question_str']
        )
        processed_question = self.text_processor({"question": question_str})
        sample.text = processed_question['token_inds']
        sample.text_len = processed_question['token_num']

        # 2. Load object
        # object bounding box information
        sample.obj_bbox_coordinates = self.copy_processor(
            {"blob": sample_info["obj_normalized_boxes"]}
        )["blob"]

        obj_tokens = [
            self.ocr_token_processor({"text": token})["text"]
            for token in sample['image_info_0']["object_tokens"]]
       #print(obj_tokens)    
        # Get FastText embeddings for OBJ tokens
        objlabel = self.obj_context_processor({"tokens": obj_tokens})
        sample.objlabel = objlabel["text"]
        sample.objlabel_tokens = objlabel["tokens"]
        sample.objlabel_tokens_enc = enc_obj2bytes(objlabel["tokens"])
        # sample.context_tokens_enc = enc_obj2bytes(context["tokens"], max_size=4094*2)
        sample.objlabel_feature_0 = objlabel["text"]
        sample.objlabel_info_0 = Sample()
        sample.objlabel_info_0.max_features = objlabel["length"]
        # 3. Load OCR
        if not self.use_ocr:
            # remove all OCRs from the sample
            # (i.e. make an empty OCR list)
            sample_info['ocr_tokens'] = []
            sample_info['ocr_info'] = []
            if 'ocr_normalized_boxes' in sample_info:
                sample_info['ocr_normalized_boxes'] = np.zeros(
                    (0, 4), np.float32
                )
            # clear OCR visual features
            sample.image_feature_1 = torch.zeros_like(sample.image_feature_1)

        # Preprocess OCR tokens
        ocr_tokens = [
            self.ocr_token_processor({"text": token})["text"]
            for token in sample_info["ocr_tokens"]
        ]
        # Get FastText embeddings for OCR tokens
        context = self.context_processor({"tokens": ocr_tokens})
        sample.context = context["text"]
        sample.context_tokens = context["tokens"]
        sample.context_tokens_enc = enc_obj2bytes(context["tokens"])
        sample.context_feature_0 = context["text"]
        sample.context_info_0 = Sample()
        sample.context_info_0.max_features = context["length"]
        # Get PHOC embeddings for OCR tokens
        context_phoc = self.phoc_processor({"tokens": ocr_tokens})
        sample.context_feature_1 = context_phoc["text"]
        sample.context_info_1 = Sample()
        sample.context_info_1.max_features = context_phoc["length"]
        # OCR order vectors
        # TODO remove order_vectors -- it is no longer needed in M4C
        order_vectors = np.eye(len(sample.context_tokens), dtype=np.float32)
        order_vectors = torch.from_numpy(order_vectors)
        order_vectors[context["length"]:] = 0
        sample.order_vectors = order_vectors
        # OCR bounding box information
        if 'ocr_normalized_boxes' in sample_info:
            # New imdb format: OCR bounding boxes are already pre-computed
            max_len = self.config.processors.answer_processor.params.max_length
            sample.ocr_bbox_coordinates = self.copy_processor(
                {"blob": sample_info['ocr_normalized_boxes']}
            )["blob"][:max_len]
        else:
            # Old imdb format: OCR bounding boxes are computed on-the-fly
            # from ocr_info
            sample.ocr_bbox_coordinates = self.bbox_processor(
                {"info": sample_info["ocr_info"]}
            )["bbox"].coordinates

        return sample

    def add_answer_info(self, sample_info, sample):
        sample_has_answer = ("answers" in sample_info)
        if sample_has_answer:
            # Load real answers from sample_info
            answers = sample_info["answers"]
            sample.gt_answers_enc = enc_obj2bytes(answers)
            answer_processor_arg = {
                "answers": answers,
                "context_tokens": sample.context_tokens,
            }
            processed_answers = self.answer_processor(answer_processor_arg)

            assert not self.config.fast_read, \
                'In M4CTextVQADataset, online OCR sampling is incompatible ' \
                'with fast_read, so fast_read is currently not supported.'
            sample.targets = processed_answers["answers_scores"]
            sample.sampled_idx_seq = processed_answers["sampled_idx_seq"]
            sample.train_prev_inds = processed_answers["train_prev_inds"]
            sample.train_loss_mask = processed_answers["train_loss_mask"]
        else:
            # Load dummy answers as placeholders
            answer_params = self.config.processors.answer_processor.params
            sample.sampled_idx_seq = None
            sample.train_prev_inds = torch.zeros(
                answer_params.max_copy_steps, dtype=torch.long
            )

        return sample
