import logging
import gensim
import json
import numpy as np
from numpy import linalg as la

logger = logging.getLogger(__name__)

def get_obj_prd_vecs(word_vector_path, dataset_path):
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        word_vector_path, binary=True)
    logger.info('Model loaded.')
    # change everything into lowercase
    all_keys = list(word2vec_model.vocab.keys())
    for key in all_keys:
        new_key = key.lower()
        word2vec_model.vocab[new_key] = word2vec_model.vocab.pop(key)
    logger.info('Wiki words converted to lowercase.')

    # if dataset_name.find('vrd') >= 0:
    with open(dataset_path + '/json_dataset/objects.json') as f:
        obj_cats = json.load(f)
    with open(dataset_path + '/json_dataset/predicates.json') as f:
        prd_cats = json.load(f)
    # elif dataset_name.find('vg') >= 0:
    #     with open(cfg.DATA_DIR + '/vg/objects.json') as f:
    #         obj_cats = json.load(f)
    #     with open(cfg.DATA_DIR + '/vg/predicates.json') as f:
    #         prd_cats = json.load(f)
    # else:
    #     raise NotImplementedError
    # represent background with the word 'unknown'
    # obj_cats.insert(0, 'unknown')
    prd_cats.insert(0, 'unknown')
    obj_cats.insert(0, 'background')
    all_obj_vecs = np.zeros((len(obj_cats), 300), dtype=np.float32)
    for r, obj_cat in enumerate(obj_cats):
        obj_words = obj_cat.split()
        for word in obj_words:
            raw_vec = word2vec_model[word]
            all_obj_vecs[r] += (raw_vec / la.norm(raw_vec))
        all_obj_vecs[r] /= len(obj_words)
    logger.info('Object label vectors loaded.')
    all_prd_vecs = np.zeros((len(prd_cats), 300), dtype=np.float32)
    for r, prd_cat in enumerate(prd_cats):
        prd_words = prd_cat.split()
        for word in prd_words:
            raw_vec = word2vec_model[word]
            all_prd_vecs[r] += (raw_vec / la.norm(raw_vec))
        all_prd_vecs[r] /= len(prd_words)
    logger.info('Predicate label vectors loaded.')
    return all_obj_vecs, all_prd_vecs


if __name__ == "__main__":
    all_obj_vecs, all_prd_vecs = get_obj_prd_vecs('/Users/pranoyr/Downloads/GoogleNews-vectors-negative300.bin','/Users/pranoyr/code/Pytorch/faster-rcnn.pytorch/data/VRD')
    print(all_obj_vecs.shape)