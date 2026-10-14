5/10/26

- Framing ML problems: frame ML problems as a function of input, output and objective function.
- classification vs regression

6/10/26

- requirements for ML systems
    - Reliability , Maintainablity , Scalability , Adaptability

7/10/26

- data architecture foundations: row Vs column major ; SQL(relational) vs NoSQL

8/10/26

- a little bit about inference engineering. Imported gpt2 and learnt it's parameters

9/10/26

- prefill VS decode phases and their pros and cons
    prefill: all input tokens processed parallely to extract attention, with one memory bus call.
    decode: each forward pass generates a single toekn output, requires a complete memory load through bus for each token generation. (autoregressive)

- reference: [The Engineering Behind LLM Inference: The Memory Wall](https://www.youtube.com/watch?v=ENkuf_2zbkc&list=PLqO45Dg1pMhlDBZTMqVL2GU-14xYip2y2&index=2)

12/10/26

- 3 parts of inference engineering: Runtime (software stack ie. CUDA, PyTorch, vLLM, etx. and model optimisation strategies) , Infrastructure (GPU and memory, networking, load balnacing and storage) , and Tooling (determining the level of abstractness to give develpers)

- Questions for requirement gathering: What model are we running inference on? Application boundaries (i/o)? Latency budget (online low latency, or offline high throughput)?,  Unit economics, expected usage patterns?

- Consumer-facing (focus on cost and flexibility) VS B2B (focus on low latency and high availability)

- Measuring inference latency & throughput for consumer facing models:
    - TTFT (time to first token)
    - TPS (tokens per second) : percieved per user / total across enterprise
    - ITL (inter token latency)

- We also measure latency in percentiles (to avoid outliers that could annoy users)
    - P50 - median latency - 1/2 requests suck
    - P90 - 90th percentile latency - 1/10 requests suck
    - P95 - 95th percentile latency - 1/20 requests suck 
    - P99 - 00th " - 1/100 requests suck 

    note: exponentially sucks less and this is what we want to measure instead of averages. 

13/10/26

- Did a little bit of hands-on : built a inference autoregressive loop step-by-step from tokenisation -> forward pass -> decoding/sampling -> loop

- A few sampling methodologies:
    - greedy : just argmax the highest probability logit (deterministic)
    - Top-k : choose randomly from among the top K probabable logits ( high K -> more random)
    - Top-p : choose smallest set of logits with probabilities that add up to P (high P -> more random)
    - Temperature : adjust the logits (using softmax, etc.) so that the probability distribution flattens 

14/10/26

- Explored GPT2 model's inner workings (without 'transformer' library):
    - extracted key model files: config.json (overview), model.safetensors (actual model weights), tokenizer files like vocab.json and merges.txt (holds merge rules)

    - In config.json :
        - model 'width' / embedding dimension / each token is represented as a vector of 768 numbers 
        - 12 transformer blocks and attention heads
        - vocabulary of ~50K tokens

    - in safetensors file:
        - SafeTensors format: 8-byte header length + JSON header + raw tensor data
        - there is token embeddings (wte), positional embeddings (wpe), attention (c_attn, c_proj), mlp/ffn (c_fn , c_proj), layer norms (ln_1, ... ln_f)
        - for each of there there is .weight and .bias file
        - FFN / MLP takes up approx 42% of the size, followed by attention and token embeddings (wte)
