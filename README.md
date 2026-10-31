# Polar
Polar is a benchmark for evaluating political bias in LLMs via comparing likelihood among options. 

## Install & SetUp
~~~bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ~
~~~

### Add Tasks to lm-eval-harness
To use Polar tasks inside lm-eval-harness, copy the task directories into your lm-eval-harness installation:
~~~bash
bash setup.sh
~~~

### Evaluation
To evaluate model on Polar, use the following command:
~~~bash
lm_eval \
    --model hf \
    --model_args pretrained={Model_Name},parallelize=True,device_map="auto",trust_remote_code=True \
    --tasks polar_us,polar_ko \
    --output_path {Directory_to_store_results} \
    --log_samples \
    --batch_size auto
~~~
The output directory should be named `results`, as the post-processing scripts assume this directory name when locating evaluation outputs.

