---
license: apache-2.0
---

# Dataset Usage

## Data Format

The dataset is provided in json format and contains the following attributes:

```json
{
    "question_id": [string] A unique identifier for the record,
    "doc_no": [string] The filename (7 digits) of the associated PDF-formatted document,
    "total_pages": [integer] The number of pages of the associated PDF-formatted document,
    "start_end_idx": [list] The start and end index of document pages used for constructing Q&A (The left and right borders included),
    "question_type": [string] The type of question, such as 'extract' for Understanding QA and 'summarize'/'count' for Reasoning QA,
    "question": [string] The question text,
    "answer": [string/float/integer/list] The correct answer for the problem,
    "detailed_evidences": [string] The detailed evidences or intermediate results in the process of Q&A construction,
    "evidence_pages": [list] The indexes of pages where correct answers can be found,
    "evidence_sources": [list] The types of evidence sources, from ['Text', 'Layout', 'Figure', 'Table'],
    "answer_format": [string] The format of concise short answer, from ['String', 'Float', 'Integer', 'List', 'None'],
    "task_tag": [string] The main task category of the Q&A pair, from ['Understanding', 'Reasoning', 'Locating'],
    "images": [list] The file paths of input images for evaluation in the main experiments in our paper,
    "pdf_path": [string] The file path of associated PDF-formatted document,
    "subTask": [list] The sub-task categories which the record belongs to, such as ["Para_Title_Locating"]
}
```

Notably, the indexes of the pages in the json file are sorted starting from 1, except for the "images" list which starts from 0.

Here is an example:

```json
{
    "question_id": "free_gpt4o_4026369_60_70_12",
    "doc_no": "4026369",
    "total_pages": 70,
    "start_end_idx": [
        60,
        70
    ],
    "question_type": "extract",
    "question": "Which publications are stated as helpful in producing a manual?",
    "answer": "University/Advantsar Communications Project",
    "detailed_evidences": "The publications stated as helpful in producing a manual are 'University/Advantsar Communications Project' published in 1997.1 and other unspecified publications (<box>(0.11, 0.41, 0.87, 0.61)</box>, page 67).",
    "evidence_pages": [
        67
    ],
    "evidence_sources": [
        "Layout"
    ],
    "answer_format": "String",
    "task_tag": "Understanding",
    "images": [
        "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_40.png",
        "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_41.png",
        ...,
        "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_68.png",
        "/data/oss_bucket_0/achao.dc/public_datasets/pdf_pngs/4000-4999/4026/4026369_69.png"
    ],
    "pdf_path": "/data/oss_bucket_0/achao.dc/public_datasets/ccpdf_zip/4000-4999/4026369.pdf",
    "subTask": [
        "SP_Layout_Understanding"
    ]
}
```

## Citation

```
@article{chao-etal-2024-longdocurl,
  author       = {Chao Deng and
                  Jiale Yuan and
                  Pi Bu and
                  Peijie Wang and
                  Zhong{-}Zhi Li and
                  Jian Xu and
                  Xiao{-}Hui Li and
                  Yuan Gao and
                  Jun Song and
                  Bo Zheng and
                  Cheng{-}Lin Liu},
  title        = {LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating
                  Understanding, Reasoning, and Locating},
  journal      = {CoRR},
  volume       = {abs/2412.18424},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2412.18424},
  doi          = {10.48550/ARXIV.2412.18424},
  eprinttype    = {arXiv},
  eprint       = {2412.18424},
  timestamp    = {Sat, 25 Jan 2025 12:51:18 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2412-18424.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```
@inproceedings{DBLP:conf/acl/DengYBWLXLGS0025,
  author       = {Chao Deng and
                  Jiale Yuan and
                  Pi Bu and
                  Peijie Wang and
                  Zhong{-}Zhi Li and
                  Jian Xu and
                  Xiao{-}Hui Li and
                  Yuan Gao and
                  Jun Song and
                  Bo Zheng and
                  Cheng{-}Lin Liu},
  editor       = {Wanxiang Che and
                  Joyce Nabende and
                  Ekaterina Shutova and
                  Mohammad Taher Pilehvar},
  title        = {LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating
                  Understanding, Reasoning, and Locating},
  booktitle    = {Proceedings of the 63rd Annual Meeting of the Association for Computational
                  Linguistics (Volume 1: Long Papers), {ACL} 2025, Vienna, Austria,
                  July 27 - August 1, 2025},
  pages        = {1135--1159},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://aclanthology.org/2025.acl-long.57/},
  timestamp    = {Thu, 24 Jul 2025 21:25:39 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/DengYBWLXLGS0025.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

Please kindly cite our paper if this paper and the codes are helpful.