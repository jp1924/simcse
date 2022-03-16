import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random

from time import localtime
import re
from datasets import load_dataset, Dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from simcse.models import RobertaForCL, BertForCL, ElectraForCL
from simcse.trainers import CLTrainer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from psutil import cpu_count

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.06,# 0.05
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="avg", # cls
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False, # False
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1, # 0.1
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False, # False
        metadata={
            "help": "Use MLP only during training"
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # ================================ data_dir ================================
    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    # ================================ data_dir ================================
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    return_tensors: bool = field(
        default=False,
        metadata={
            "help":"토크나이징시 pytorch_tensor로 변환 할지 말지를 결정하는 부분"
            "pytorch_tensor로 return을 할 때 무조건 pad_to_max_length를 True로 만들어 줘야함."
            "그렇지 않으면 에러가 발생함."
        }
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    # ========================= custom_section =========================    

    output_dir: str = field(
        default=None
    )
    num_train_epochs: int = field(
        default=9 # best score: 9
    )
    per_device_train_batch_size: int = field(
        default=512 # 256
    )
    learning_rate: float = field(
        default=3e-5
    )
    eval_original: bool = field(
        default= False,
        metadata={"help":"eval_bi와 eval_cross 둘다 False라면 작동하지 않음."
        "false이여야 custom evaluation라인이 작동함."
        }
    )
    eval_bi: bool = field(
        default=True,
        metadata={"help":"eval_original이 False라면 작동하지 않는다."}        
    )
    eval_cross: bool = field(
        default=True,
        metadata={"help":"eval_original이 False라면 작동하지 않는다."}    
    )
    do_train: bool = field(
        default=True
    )
    do_eval: bool = field(
        default=True
    )
    eval_steps: int = field(
        default=125
    )
    do_predict: bool = field(
        default=False
    )
    fp16: bool = field(
        default=True
    )
    fp16_full_eval: bool = field(
        default=False
    )
    no_cuda: bool = field(
        default= False
    )
    metric_for_best_model: str = field(
        default="stsb_spearman"
    )
    logging_strategy: str = field(
        default="steps"
    )
    logging_steps: int = field(
        default=1
    )
    save_steps: int = field(
        default=1000000
    )
    pre_evaluation: bool = field(
        default=False,
        metadata={"help": """
        주의사항: 사전 eval을 진행하면 fp16환경에서 정상적으로 진행이 안될 수 있다. 이걸 작성하고 있는 시점에서 아직 해결하지 못함.
        """}
    )
    original_train_data_load: bool = field(
        default=False
    )

    dataloader_drop_last: bool = field(
        default=True
    )# drop_last를 하지 않으면 학습 중간에 
     # Input tensor at index 3 has invalid shape [2, 2], but expected [2, 5]
     # 와 같은 에러가 발생함.
     # 이유 : 254 batch로 들어가던 데이터가 마지막에 17batch로 들어가서 생기는 에러.
    # ========================= custom_section =========================

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
        model_name = "beomi/kcbert-base" if False else "monologg/koelectra-base-v3-discriminator"

        model_args.model_name_or_path = model_name\
            if model_args.model_name_or_path == None \
                else model_args.model_name_or_path
        import re
        t = {re.sub("tm_*", "", x):getattr(localtime(), x) for x in dir(localtime()) if re.match("tm_*", x) != None}
        model_name = "-".join(model_args.model_name_or_path.split("/"))
        DIR_NAME = f"""[unsup]SimCSE/[{t["mon"]}-{t["mday"]}-{t["hour"]}:{t["min"]}:{t["sec"]}]SimCSE-{model_name}"""
        OUT_DIR = os.path.abspath(f"""workspace/{DIR_NAME}""")
        
        training_args.output_dir = OUT_DIR \
            if OUT_DIR == None \
                else training_args.output_dir + f"/{DIR_NAME}"

        print(f"""\n{training_args.output_dir}\n""")
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )



    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if training_args.original_train_data_load:
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        if extension == "csv":
            datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
        else:
            datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")
    # ========================= custom_section =========================

    else:
        import re
        regex = lambda x: {"text":re.sub(r'(\\t)|(\\r)|(\\n)|(\[\')|(\'\])|(\\)', "", x["text"]).strip()}
        
        #데이터를 불러오는 부분.
        if data_args.train_file:
            loaded_data = Dataset.from_text(data_args.train_file)
        else:
            loaded_data = Dataset.from_text(os.path.abspath("/data/lawtime_passage.txt"))

        refine_data = loaded_data.map(regex, num_proc=cpu_count())
        datasets = refine_data.train_test_split(test_size=0.1, train_size=0.9)

    # ========================= custom_section =========================

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:

        model_name = model_args.tokenizer_name
        # ========================= custom_section =========================

        if "kobert" in model_name and "monologg" in model_name:
            from tokenization_kobert import KoBertTokenizer
            tokenizer = tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
            print("\n============== SimCSE : [monologg/kobert] ==============\n")
        elif "kobert" in model_name and "skt" in model_name:
            from kobert import get_tokenizer
            from gluonnlp.data import SentencepieceTokenizer
            tokenizer = SentencepieceTokenizer(get_tokenizer())
            print("\n============== SimCSE : [skt/kobert] ==============\n")

        # ========================= custom_section =========================
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:

        model_name = model_args.model_name_or_path

        # ========================= custom_section =========================

        if "kobert" in model_name and "monologg" in model_name:
            from tokenization_kobert import KoBertTokenizer
            tokenizer = tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
            print("============== SimCSE : Select [monologg/kobert] Cross_model ==============")

        # ========================= custom_section =========================

        else:

            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        if 'roberta' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args                  
            )
        elif 'bert' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
            if model_args.do_mlm:
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        elif 'electra' in model_args.model_name_or_path:
            model = ElectraForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Prepare features
    column_names = datasets["train"].column_names
    sent2_cname = None
    if len(column_names) == 2:
        # Pair datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
    elif len(column_names) == 3:
        # Pair datasets with hard negatives
        sent0_cname = column_names[0]
        sent1_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 1:
        # Unsupervised datasets
        sent0_cname = column_names[0]
        sent1_cname = column_names[0]
    else:
        raise NotImplementedError

    # 데이터 encoding부분
    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields , 필터링 거치는 부분
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
            if examples[sent1_cname][idx] is None:
                examples[sent1_cname][idx] = " "
        # 여기서 input된 데이터를 두개로 만듬.
        sentences = examples["text"]
        # sentences = examples[sent0_cname] + examples[sent1_cname]
        # ^^^^^^^^^^^^^^^^^^^^^ 되돌려 놓을 것 ^^^^^^^^^^^^^^^^^^^^^


        # ========================= custom_section =========================

        if ModelArguments.do_mlm:
            # masking을 하는 부분. 전체 단어중 15%의 비율로 masking을 함.
            import random
            for idx_y, train_data in enumerate(sentences):

                train_data = train_data.split(' ')
                word_num = len(train_data)

                count_max = int(word_num * 0.1215) if isinstance(word_num * 0.1215, float) else word_num * 0.1215

                for count_y, idx_x in enumerate(random.sample(range(0, word_num-1), count_max)):

                    train_data[idx_x] = tokenizer._mask_token

                sentences[idx_y] = ' '.join(train_data)

        # ========================= custom_section =========================

        # If hard negative exists
        if sent2_cname is not None:
            for idx in range(total):
                if examples[sent2_cname][idx] is None:
                    examples[sent2_cname][idx] = " "
            sentences += examples[sent2_cname]

        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
            return_tensors= "pt" if data_args.return_tensors else None
        )

        features = {}
        if sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                # features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
                # ^^^^^^^^^^^^^^^^^^^^^ 되돌려 놓을 것 ^^^^^^^^^^^^^^^^^^^^^
                features[key] = [[sent_features[key][i], sent_features[key][i]] for i in range(total)]
                # [key][i+total] -> [key][i]

                # 이 부분이 sentence pair를 만드는 부분
        # ========================= 방법 =========================
        # 1. 동일한 데이터 셋 2개를 만든 다음 이들을 concat함 ex : 1000개 데이터가 있다면 1000 + 1000개를 붙여 2000개로 만듬
        # 2. 붙인 데이터를 tokenizer에다가 넣어서 encoding시킴
        # 3. encoding시킨 데이터가 2000개 라고 했을 때 0번과 1000번 데이터를 서로 붙임
        # 4. 이걸 각 데이터마다 진행시켜줌
        # =========================== ? ===========================
        # [[sent_features[key][i], sent_features[key][i+total]] 에서 sent_features[key][i+total]와 같이 하지말고 차라리
        # [[sent_features[key][i], sent_features[key][i]]를 하면 되지 않나? 그럼 동일한 데이터를 두번 붙여서 n*2로 만들 필요도 없고
        # tokenizing을 진행하면서 속도도 더 빨라질거같은데.....
        return features

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            # num_proc=cpu_count(),
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    @dataclass
    class OurDataCollatorWithPadding:

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        mlm: bool = True
        mlm_probability: float = data_args.mlm_probability

        def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
            bs = len(features)
            if bs > 0:
                num_sent = len(features[0]['input_ids'])
            else:
                return
            flat_features = []
            for feature in features:
                for i in range(num_sent):
                    flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

            batch = self.tokenizer.pad(
                flat_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            if model_args.do_mlm:
                batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

            batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            return batch
        
        def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
            """
            inputs = inputs.clone()
            labels = inputs.clone()
            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            return inputs, labels

    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)

    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.model_args = model_args


    # ========================= custom_section =========================

    if training_args.pre_evaluation:
        
        logger.info("*** Evaluate ***")
        result_list = trainer.evaluate(eval_senteval_transfer=True)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for results in result_list:
                    for key, value in sorted(results.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

    #Caution: cuda를 true, fp16을 True로 한 상태에서 사전 evaluate를 진행하면 에러가 발생한다.
    # 아마  trainer.py 167번줄의 encode[k] = encode[k].to(f"cuda:{device_idx}") 때문에 발생하는 문제인거 같다.

    # ========================= custom_section =========================

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
            else None
        )
        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        result_list = trainer.evaluate(eval_senteval_transfer=True)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for results in result_list:
                    for key, value in sorted(results.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # ========================= custom_section =========================

    import setproctitle
    setproctitle.setproctitle('[SimCSE]test')
    os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'

    # ========================= custom_section =========================
    main()
