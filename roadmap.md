# 2-Month ML System Design & Inference Engineering Roadmap
**Study Schedule:** 6 Days/Week (Mon–Sat) | **Month 1:** 1 Hour/Day | **Month 2:** 2 Hours/Day
**Tracks:** Mon–Wed: ML Systems Design (Chip Huyen) | Thu–Sat: Practical Inference Engineering & LLM Optimization

---

## Month 1: Foundations of ML Systems & Single-Instance LLM Inference (1 Hr/Day)

---

### Week 1: ML Systems Principles & LLM Generation Mechanics
* **Weekly Focus:** Shifting from academic ML to production constraints, defining spec-driven MVPs, and mastering the low-level autoregressive text generation loop.
* **Weekly Goal:** Be able to scope a real-world ML project with strict latency/SLA boundaries and trace an LLM request from raw token input to logits and decode loops.
* **3 End-of-Week Master Questions:**
  1. *How does the transition from training to online inference fundamentally alter the primary computational bottleneck of a transformer model?*
  2. *In a production system facing strict SLA latency constraints (e.g., p95 < 150ms), why is decoupling business metrics from model loss functions essential for system adaptability?*
  3. *Why does the memory layout of an autoregressive decode loop make standard batching inefficient on modern GPUs without specialized execution engines?*

#### Day 01 (Mon): Chip Huyen Ch 1 — Overview of ML Systems
* **Topic:** Research vs. Production, Stakeholder Alignment, & System Requirements
* **3 Senior MLE Practice Questions:**
  1. **Q1 (System Trade-offs):** A business team insists that raising offline model accuracy from 92% to 95% is critical. However, achieving this requires a 4x larger ensemble that increases p99 latency from 80ms to 450ms. As the Lead MLE, how do you evaluate and communicate the system-level trade-offs?
  2. **Q2 (Failure Modes):** Why is a model that serves low-confidence, plausibly wrong predictions much more dangerous to an e-commerce platform than an API endpoint returning a `504 Gateway Timeout`?
  3. **Q3 (Production Constraints):** How do data compliance regulations (e.g., GDPR right-to-be-forgotten) impact the maintainability pillar of an ML system that relies on continuous retraining?

#### Day 02 (Tue): Chip Huyen Ch 2 — ML Systems Design Process
* **Topic:** Business Objectives vs. ML Metrics, Framing Tasks, & Objective Decoupling
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Problem Framing):** An e-commerce app wants to recommend products to users. Explain why framing this as a regression task (predicting engagement score for a user-item pair) scales better in production than framing it as a multi-class classification task over all product SKUs.
  2. **Q2 (Objective Misalignment):** A news aggregator trains a model to maximize click-through rate (CTR). After deployment, user retention drops by 20%. What underlying phenomenon occurred, and how do you redesign the loss/scoring function to fix it?
  3. **Q3 (Decoupled Architectures):** Why is it better to deploy two separate models—one predicting click probability $P(\text{click})$ and one predicting conversion value $V(\text{conv})$—and combine their outputs at serving time ($Score = \alpha \cdot P(\text{click}) + \beta \cdot V(\text{conv})$), rather than training a single unified multi-task model?

#### Day 03 (Wed): Chip Huyen Ch 2 — Spec-Driven Scoping & MVP Architecture
* **Topic:** Iterative Development, Spec Files, & Minimizing Technical Uncertainty
* **3 Senior MLE Practice Questions:**
  1. **Q1 (MVP Design):** A team wants to build a real-time voice-to-structured-medical-record assistant. What is the minimal CLI MVP you would build in Week 1 to test core technical viability before writing any UI or orchestration code?
  2. **Q2 (Spec Engineering):** In a spec-driven development workflow, what specific fields must be defined in the schema for input data validation, and why should schema validation happen at the API edge rather than inside the model worker?
  3. **Q3 (Ablation & Baselines):** Why is establishing a simple heuristic baseline (e.g., "popular items in user's zip code") required before evaluating any deep learning model during initial scoping?

#### Day 04 (Thu): Day 01 — LLM Inference Mechanics: End-to-End Text Generation
* **Topic:** Prefill vs. Decode Phase Mechanics & Token Generation Loop (Inference Eng Ch 2.2)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Compute vs. Memory Bottleneck):** Why is the Prefill phase of LLM inference compute-bound (high arithmetic intensity), whereas the Decode phase is memory-bandwidth bound (low arithmetic intensity)?
  2. **Q2 (Latency Decomposition):** How do Time-To-First-Token (TTFT) and Inter-Token Latency (ITL) map to the Prefill and Decode phases, and which model parameters dictate each metric?
  3. **Q3 (Sampling Mechanics):** How does temperature scaling alter the logit distribution before softmax, and why does greedy decoding ($T=0$) eliminate output variance while increasing vulnerability to repetition loops?

#### Day 05 (Fri): Day 02 — Inference from Scratch: Model Internals & Tokenization
* **Topic:** Tokenizer Mechanics, Vocabulary, BPE, & Sampling (Inference Eng Ch 2.2)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Tokenizer Edge Cases):** A multi-lingual customer support model fails when handling mixed English and character-heavy languages (e.g., Japanese). How does Byte-Pair Encoding (BPE) vocabulary size affect tokenization density and context window consumption for non-English text?
  2. **Q2 (Security & Sanitation):** How can adversarial prompt injection exploit tokenizer artifacts (e.g., special tokens like `<|im_start|>`), and where in the inference pipeline must token sanitization occur?
  3. **Q3 (Execution Profiling):** If your tokenizer runs on the host CPU in Python while model inference runs on the GPU, how can host-to-device token transfer become an unexpected bottleneck during high-throughput batching?

#### Day 06 (Sat): Day 03 — Embeddings: From Integers to Vectors
* **Topic:** Token Embeddings, Positional Encodings, & Rotary Position Embedding (RoPE) (Inference Eng Ch 2.2.1)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Position Encoding Trade-offs):** Why do absolute positional encodings fail to extrapolate to sequences longer than those seen during training, whereas relative encodings like RoPE scale better?
  2. **Q2 (Rotary Embeddings Mechanics):** How does RoPE inject positional information into Query and Key vectors via complex rotation matrices without modifying Value vectors?
  3. **Q3 (Memory Access Patterns):** When computing embedding lookups for a batch of input tokens, why are embedding operations heavily bound by GPU memory bandwidth rather than compute unit utilization?

---

### Week 2: Data Pipelines & Transformer Attention Internals
* **Weekly Focus:** Storage engines, streaming data pipelines, transformer attention mechanics, and the arithmetic intensity of KV caching.
* **Weekly Goal:** Architect reliable data ingestion layers and calculate exact memory and bandwidth requirements for transformer inference workloads.
* **3 End-of-Week Master Questions:**
  1. *How does the choice between row-oriented and column-oriented storage engines dictate online feature lookup latency vs. offline batch training throughput?*
  2. *Given a model with $L$ layers, $H$ heads, head dimension $D$, and precision $P$ bytes, what is the exact formula for KV cache memory growth per token per request?*
  3. *Using the Roofline Model, how do you determine whether a specific layer in an LLM inference pass is bound by GPU memory bandwidth or Tensor Core TFLOPS?*

#### Day 07 (Mon): Chip Huyen Ch 3 — Data Engineering: Storage Engines
* **Topic:** Relational vs. NoSQL, Columnar vs. Row Storage, & Vector Databases
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Storage Architecture):** Why would you select a key-value store (e.g., Redis/Cassandra) for serving real-time user features at inference, but use a columnar database (e.g., ClickHouse/Snowflake) for generating training sets?
  2. **Q2 (Vector DB System Design):** In a real-time Retrieval-Augmented Generation (RAG) system, what are the trade-offs between Approximate Nearest Neighbor (ANN) index types (HNSW vs. IVF-PQ) in terms of recall, memory footprint, and index build time?
  3. **Q3 (Data Consistency):** How do you prevent race conditions when online prediction nodes read features that are concurrently being updated by a streaming ingestion job?

#### Day 08 (Tue): Chip Huyen Ch 3 — Data Processing Paradigms
* **Topic:** ETL vs. ELT, Batch vs. Stream Processing, & Event-Driven Architecture
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Streaming vs. Batch):** An online fraud detection system requires feature updates within 500ms of a transaction. Why is a traditional daily batch ETL pipeline insufficient, and how does a streaming architecture (Kafka + Flink) solve this?
  2. **Q2 (Watermarking & Late Data):** In a stream processing pipeline calculating rolling 5-minute user activity counts, how do you handle late-arriving events caused by network delays on client devices?
  3. **Q3 (Idempotency):** Why must event processing in streaming ML feature pipelines be idempotent, and how do duplicate events distort feature distribution metrics?

#### Day 09 (Wed): Chip Huyen Ch 3 — Storage/Compute Separation
* **Topic:** Kafka, Message Brokers, & Decoupled Data Systems
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Queue Backpressure):** During a flash sale, incoming inference requests spike 10x. How does a message queue (e.g., Kafka) protect downstream inference workers from crashing due to sudden load spikes?
  2. **Q2 (Consumer Group Scaling):** If your inference pipeline is bottlenecked by feature extraction from Kafka messages, how do you scale consumer groups without causing partition rebalancing storms?
  3. **Q3 (Data Retention Policies):** What strategy do you implement for log retention when raw inference request logs are used for offline model drift monitoring vs. immediate real-time debugging?

#### Day 10 (Thu): Day 04 — Transformer Blocks & Attention Deep Dive
* **Topic:** Multi-Head Attention (MHA), Grouped-Query Attention (GQA), & Scaled Dot-Product (Inference Eng Ch 2.2.2–2.2.3)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Attention Architecture):** Why has the industry shifted from Multi-Head Attention (MHA) to Grouped-Query Attention (GQA) in open LLM architectures (like Llama 3)? Calculate the KV cache reduction ratio when moving from MHA (32 KV heads) to GQA (8 KV heads).
  2. **Q2 (Attention Matrix Memory):** Why does standard scaled dot-product attention require $O(N^2)$ memory for sequence length $N$, and how does FlashAttention avoid materializing the full $N \times N$ attention matrix in GPU High Bandwidth Memory (HBM)?
  3. **Q3 (Softmax Numerical Stability):** Why is subtractive normalization ($	ext{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum e^{x_j - \max(x)}}$) mandatory in GPU kernel implementations of attention?

#### Day 11 (Fri): Day 05 — KV Cache Mechanics
* **Topic:** Key-Value Cache, Memory Footprint Calculations, & Context Growth (Inference Eng Ch 2.2)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (KV Cache Sizing):** Calculate the exact KV cache footprint in Gigabytes for a 70B parameter model (80 layers, 8 KV heads, head dimension 128, FP16 precision) serving a batch of 64 requests with a sequence length of 4,096 tokens.
  2. **Q2 (Memory Fragmentation):** Why does naive contiguous memory allocation for KV cache cause severe external memory fragmentation in production LLM serving engines?
  3. **Q3 (Trade-off Analysis):** If a GPU has 80GB VRAM and the 70B model weights occupy 140GB across 2 GPUs (70GB per GPU), how much memory remains on each GPU for KV cache and activation memory?

#### Day 12 (Sat): Day 06 — Ops:Byte Ratio & Arithmetic Intensity
* **Topic:** Roofline Model, Memory-Bandwidth vs. Compute Bound (Inference Eng Ch 2.4)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Arithmetic Intensity Calculation):** An NVIDIA H100 GPU has 1,979 TFLOPS of FP16 tensor performance and 3.35 TB/s HBM3 memory bandwidth. What is the operational intensity threshold (FLOPs/byte) above which a kernel becomes compute-bound?
  2. **Q2 (Bottleneck Diagnosis):** During single-request LLM generation (batch size = 1), a model layer performs $2 \times 10^9$ FLOPs and reads $1 \times 10^9$ bytes of weights. Is this pass compute-bound or memory-bound on an H100 GPU?
  3. **Q3 (Batch Size Impact):** How does increasing the inference batch size shift the operational intensity of matrix multiplication ($Y = XW$) from memory-bound to compute-bound?

---

### Week 3: Data Curation & Modern Serving Frameworks
* **Weekly Focus:** Data labeling, weak supervision, kernel fusion, ONNX/TensorRT export, and vLLM PagedAttention mechanics.
* **Weekly Goal:** Build robust training/evaluation datasets and understand how paged virtual memory transforms LLM serving throughput.
* **3 End-of-Week Master Questions:**
  1. *How does programmatic labeling (weak supervision) mitigate class imbalance while introducing correlated label noise into evaluation sets?*
  2. *What hardware mechanisms allow CUDA kernel fusion to dramatically speed up memory-bandwidth-bound activation and norm layers?*
  3. *How does vLLM's PagedAttention eliminate memory fragmentation and enable dynamic memory sharing during LLM inference?*

#### Day 13 (Mon): Chip Huyen Ch 4 — Training Data: Sampling & Curation
* **Topic:** Reservoir Sampling, Stratified Sampling, & Active Learning
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Sampling Bias):** You are building a model to predict rare credit card fraud (0.01% incidence). If you undersample non-fraud transactions to achieve a 50/50 training split, how must you recalibrate the model's raw probability output at inference time?
  2. **Q2 (Active Learning Selection):** What are the trade-offs between uncertainty sampling (margin sampling) and diversity sampling when selecting unlabelled production data points for human annotation?
  3. **Q3 (Reservoir Sampling in Streams):** How do you implement reservoir sampling over an infinite stream of user interaction logs to maintain a uniform random sample of size $K$ without knowing total stream length?

#### Day 14 (Tue): Chip Huyen Ch 4 — Data Labeling & Weak Supervision
* **Topic:** Programmatic Labeling, Snorkel, & Labeling Rubrics
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Weak Supervision Architecture):** How does a generative label model (e.g., Snorkel) combine noisy labeling functions with different accuracies and correlations without ground truth labels?
  2. **Q2 (Labeling Rubric Drift):** When human annotators show low inter-annotator agreement (e.g., Fleiss' Kappa < 0.4) on an intent classification task, is this an annotation quality issue or an ambiguous task framing problem? How do you resolve it?
  3. **Q3 (Semi-Supervised Pseudo-Labeling):** What failure modes occur when using a teacher model to generate pseudo-labels for unlabeled production logs to train a smaller student model?

#### Day 15 (Wed): Chip Huyen Ch 4 — Label Consistency & Data Quality
* **Topic:** Class Imbalance, Synthetic Data, & Data Validation
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Synthetic Data Risks):** Using an LLM to generate synthetic training data for a domain-specific classifier can introduce mode collapse and hallucinated patterns. How do you design a validation filter for synthetic samples?
  2. **Q2 (Class Imbalance Metrics):** Why is Accuracy a misleading metric for highly imbalanced classification tasks, and under what conditions should you prioritize PR-AUC over ROC-AUC?
  3. **Q3 (Data Schema Drift):** How do automated schema validation checks (e.g., Great Expectations) integrated into CI/CD prevent malformed features from reaching training pipelines?

#### Day 16 (Thu): Day 07 — CUDA Kernels, Kernel Selection & Kernel Fusion
* **Topic:** GPU Execution Model, Threads, Blocks, Warps, & Kernel Fusion (Inference Eng Ch 4.1)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (GPU Memory Hierarchy):** Compare the latency and bandwidth of GPU Registers, Shared Memory (SRAM), and High Bandwidth Memory (HBM). Why is kernel fusion (e.g., fusing LayerNorm + GELU) so effective at reducing HBM round-trips?
  2. **Q2 (Warp Divergence):** What is warp divergence in a CUDA kernel, and how does conditional branching (`if/else`) inside a custom kernel degrade GPU compute utilization?
  3. **Q3 (Shared Memory Bank Conflicts):** How do bank conflicts occur in GPU shared memory, and how do CUDA developers pad data structures to prevent serialized memory accesses?

#### Day 17 (Fri): Day 08 — PyTorch, File Formats, ONNX & TensorRT
* **Topic:** Model Serialization, ONNX Graph Representation, & TensorRT Compilation (Inference Eng Ch 4.2)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Graph Compilation):** Why is executing PyTorch models in eager mode inefficient for production deployment, and what optimizations does a compiled execution graph (via ONNX / TensorRT) perform?
  2. **Q2 (Dynamic Shapes):** How do dynamic input shapes (varying sequence lengths) impact TensorRT engine optimization, and what are optimization profiles?
  3. **Q3 (Safe Serialization):** Why has the industry abandoned PyTorch `.pt` / `.bin` pickle files in favor of `.safetensors` for production weight distribution?

#### Day 18 (Sat): Day 09 — vLLM: PagedAttention & Continuous Batching
* **Topic:** Virtual Memory for KV Cache, Memory Fragmentation, & Continuous Batching (Inference Eng Ch 4.3.1)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (PagedAttention Mechanics):** How does PagedAttention map non-contiguous physical GPU memory blocks to logical KV cache blocks, similar to virtual memory paging in operating systems?
  2. **Q2 (Continuous Batching vs. Static Batching):** Explain why traditional static batching wastes GPU compute due to padding tokens when sequence lengths vary, and how iteration-level continuous batching eliminates this overhead.
  3. **Q3 (Throughput Impact):** If vLLM increases memory utilization from 20% (static allocation) to 96% (paged allocation), how does this directly scale maximum serving concurrency on a single GPU?

---

### Week 4: Feature Engineering & Advanced Inference Engines
* **Weekly Focus:** Feature leakage prevention, feature stores, SGLang RadixAttention, TensorRT-LLM, and disaggregated serving basics.
* **Weekly Goal:** Eliminate subtle training pipeline bugs and master modern LLM execution engine architectures.
* **3 End-of-Week Master Questions:**
  1. *How does temporal data leakage occur during feature extraction, and how do point-in-time joins resolve it in production feature stores?*
  2. *How does RadixAttention in SGLang utilize a radix tree structure to enable automatic KV cache sharing across multi-turn chats and prefix prompts?*
  3. *Why does disaggregating the Prefill and Decode phases onto separate GPU nodes resolve the resource contention caused by differing arithmetic intensities?*

#### Day 19 (Mon): Chip Huyen Ch 5 — Feature Engineering Operations
* **Topic:** Scaling, Encoding, Feature Crossing, & Dimensionality Reduction
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Categorical Encoding in Prod):** Why can One-Hot Encoding cause production pipeline failures when encountering unseen categorical values at inference, and why is Hash Encoding a safer alternative for high-cardinality features?
  2. **Q2 (Feature Crossing Mechanics):** When should you explicitly create feature crosses (e.g., `User_Country x Item_Category`) for linear/tree models versus relying on deep neural networks to learn implicit feature interactions?
  3. **Q3 (Scaling & Normalization Shift):** If a feature is normalized using training set mean and variance ($\mu, \sigma$), where and how must those parameters be stored to ensure identical transformation during online inference?

#### Day 20 (Tue): Chip Huyen Ch 5 — Data Leakage & Time-Series Splitting
* **Topic:** Target Leakage, Temporal Leakage, & Validation Strategies
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Temporal Leakage):** A model predicting user churn achieves 99% accuracy offline but fails in production. You discover the training set used random k-fold cross-validation instead of temporal splitting. Why did this cause leakage?
  2. **Q2 (Group Leakage):** When building a medical diagnostic model from patient X-ray images, why must data splits be grouped by `Patient_ID` rather than randomly partitioning individual images?
  3. **Q3 (Feature Leakage Auditing):** How do you build an automated pipeline check that computes feature-target correlations to flag potential leakage before model training begins?

#### Day 21 (Wed): Chip Huyen Ch 5 & 10 — Feature Stores
* **Topic:** Feature Store Architecture, Online vs. Offline Sync, & Point-in-Time Joins
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Point-in-Time Joins):** How does a Feature Store (e.g., Feast) execute point-in-time spatial-temporal joins to ensure offline training data reflects exact historical feature states without future bias?
  2. **Q2 (Online-Offline Skew):** What architectural pattern prevents online-offline feature skew when feature transformations are written in Python for offline training but re-implemented in C++/Java for low-latency online serving?
  3. **Q3 (Storage Backend Selection):** Why do feature stores use low-latency key-value stores (Redis/DynamoDB) for online reads and object store data lakes (S3/Parquet) for offline reads?

#### Day 22 (Thu): Day 10 — SGLang: RadixAttention & Structured Outputs
* **Topic:** Radix Tree KV Cache Reuse, Context Sharing, & Constrained Decoding (Inference Eng Ch 4.3.2)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (RadixAttention Mechanics):** How does SGLang maintain a radix tree over KV cache blocks in GPU memory, and what happens to the tree during memory pressure when LRU eviction triggers?
  2. **Q2 (Prefix Cache Reuse Efficiency):** In a multi-turn conversational AI system, how does RadixAttention reduce TTFT for turn $N$ by reusing the KV cache generated in turns $1 \dots N-1$?
  3. **Q3 (Structured Output Execution):** How do constrained decoding engines (e.g., Outlines, XGrammar) enforce JSON schemas during LLM generation by masking invalid tokens at the logit level before sampling?

#### Day 23 (Fri): Day 11 — TensorRT-LLM: Compilation & Plugin System
* **Topic:** Graph Compilation, Custom C++ Plugins, & High-Throughput Execution (Inference Eng Ch 4.3.3)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (TensorRT-LLM Optimization):** How does TensorRT-LLM optimize execution graphs through operator fusion, FP8 precision calibration, and custom GEMM kernel selection?
  2. **Q2 (C++ Plugin Architecture):** When an LLM model architecture includes a novel attention mechanism not natively supported by TensorRT, how do you write and integrate a custom C++/CUDA plugin?
  3. **Q3 (Build Time vs. Runtime Trade-off):** Why does TensorRT-LLM engine compilation take tens of minutes per GPU target architecture, and how do you manage compiled engine artifacts in a production deployment pipeline?

#### Day 24 (Sat): Day 12 — NVIDIA Dynamo: Disaggregated Serving
* **Topic:** Prefill/Decode Disaggregation & Inter-Node KV Cache Transfer (Inference Eng Ch 4.4)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Disaggregation Motivation):** Why does running Prefill and Decode on the same GPU instance lead to head-of-line blocking and degraded p99 latency for short incoming requests during heavy prefill bursts?
  2. **Q2 (KV Cache Transport):** In a disaggregated serving architecture (e.g., NVIDIA Dynamo), what network technologies (e.g., NVLink, RDMA over Converged Ethernet / RoCE) are required to transfer KV cache tensors from Prefill nodes to Decode nodes with sub-millisecond overhead?
  3. **Q3 (Load Balancing Ratio):** How do you dynamically compute the optimal pool ratio of Prefill GPU instances to Decode GPU instances based on incoming request prompt lengths and generation lengths?

---

## Month 2: Distributed Systems, Advanced Quantization, & MLOps Infrastructure (2 Hours/Day)

---

### Week 5: Advanced LLM Optimizations & Distributed Baselines
* **Weekly Focus:** Quantization algorithms (GPTQ, AWQ, SmoothQuant), speculative decoding, prefix routing, and distributed model selection.
* **Weekly Goal:** Reduce model precision to FP8/INT4 with minimal quality degradation and accelerate generation using speculative draft models.
* **3 End-of-Week Master Questions:**
  1. *How does SmoothQuant resolve activation outliers in transformer models by mathematically shifting quantization difficulty from activations to weights?*
  2. *What is the mathematical condition under which Speculative Decoding guarantees identical output distributions to the target model while achieving >2x latency speedups?*
  3. *How does cache-aware routing steer incoming requests to specific GPU workers to maximize prefix cache hits across a multi-tenant cluster?*

#### Day 25 (Mon): Chip Huyen Ch 6 — Model Selection & Baselines
* **Topic:** Zero-Rule & Heuristic Baselines, Ensembling, & Model Complexity
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Baseline Selection):** When building a complex fraud detection deep neural network, why must you simultaneously evaluate a simple GBDT baseline (e.g., XGBoost)? Under what operational conditions would you deploy the GBDT over the neural network?
  2. **Q2 (Ensemble Serving Latency):** An ensemble of 5 distinct models increases prediction accuracy by 2%. However, running all 5 models sequentially breaks SLA limits. How can you parallelize or distill this ensemble for real-time serving?
  3. **Q3 (Model Stacking Architecture):** In a multi-stage ranking system (Retrieval -> Scoring -> Re-ranking), how do you prevent errors from propagating and compounding through the pipeline stages?

#### Day 26 (Tue): Chip Huyen Ch 6 — Distributed Training Paradigms
* **Topic:** Data Parallelism (DDP), Fully Sharded Data Parallel (FSDP), & Pipeline/Tensor Parallelism
* **3 Senior MLE Practice Questions:**
  1. **Q1 (DDP vs. FSDP):** Explain why Distributed Data Parallel (DDP) duplicates full model parameters across every GPU and crashes due to OOM when training 70B models, whereas FSDP shards weights, gradients, and optimizer states across rank workers.
  2. **Q2 (All-Reduce Communication):** How does Ring-AllReduce reduce network communication overhead during distributed backpropagation, and why is inter-node network bandwidth (e.g., InfiniBand vs. Ethernet) the main scaling bottleneck?
  3. **Q3 (Gradient Accumulation):** When hardware memory constraints limit your per-GPU batch size to 2, how do you use gradient accumulation to simulate an effective global batch size of 1,024 without extra VRAM?

#### Day 27 (Wed): Chip Huyen Ch 6 — Offline Evaluation & Slice-Based Testing
* **Topic:** Subgroup Analysis, Behavioral Testing, & Error Analysis
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Slice-Based Failures):** A sentiment analysis model achieves 95% aggregate F1 score across a general benchmark set but drops to 45% on text containing negation idioms ("not terrible"). How do you construct automated slice-based tests to prevent deploying such regressions?
  2. **Q2 (Behavioral Testing Types):** Give concrete examples of Invariance (INV) tests and Directional Expectation (DIR) tests for a credit scoring ML pipeline.
  3. **Q3 (Gold-Standard Test Set Governance):** Why must evaluation test sets be strictly isolated from development environments, and how do you detect test set contamination in open-weight LLMs?

#### Day 28 (Thu): Day 13 & 14 — Quantization: Formats & Algorithms
* **Topic:** FP8, INT8, INT4, NVFP4, PTQ vs. QAT, GPTQ, AWQ, & SmoothQuant (Inference Eng Ch 5.1.1–5.1.2)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Quantization Mechanics):** Derive the linear quantization formula mapping a continuous floating-point range $[x_{min}, x_{max}]$ to an 8-bit integer range $[-128, 127]$ given scale factor $S$ and zero-point $Z$.
  2. **Q2 (AWQ vs. GPTQ):** How does Activation-aware Weight Quantization (AWQ) protect salient weight channels based on activation magnitudes, and why does this outperform rounding-error-only quantization methods like GPTQ on lower bit-widths (INT4)?
  3. **Q3 (FP8 Formats):** Compare NVIDIA's FP8 formats: E4M3 (4 exponent, 3 mantissa) vs. E5M2 (5 exponent, 2 mantissa). Which format is optimal for the forward pass activation matrices vs. gradients/KV cache?

#### Day 29 (Fri): Day 15 — Speculative Decoding
* **Topic:** Draft-Target Models, Acceptance Criterion, Medusa, & EAGLE (Inference Eng Ch 5.2)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Speculative Decoding Proof):** How does the modified rejection sampling criterion in speculative decoding ensure that the accepted token sequence follows the exact probability distribution of the large target model?
  2. **Q2 (Draft Model Selection):** If a draft model takes $2ms$ per token and has a $70\%$ acceptance rate per token on a $50ms/token$ target model, what is the theoretical speedup factor for generating a sequence of length $N$?
  3. **Q3 (Medusa/EAGLE Architecture):** How do multi-head speculative architectures (like Medusa) eliminate the need for a separate draft model by appending extra prediction heads directly to the base transformer backbone?

#### Day 30 (Sat): Day 16 — KV Cache: Prefix Caching & Cache-Aware Routing
* **Topic:** Prefix Caching, Dynamic Cache Routing, & Multi-Tenant Isolation (Inference Eng Ch 5.3)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Cache-Aware Load Balancing):** You run a cluster of 8 vLLM GPU nodes. An incoming request arrives with a system prompt that matches an existing KV cache block stored on GPU Node 3. How does a cache-aware router route this request to balance cache reuse vs. queue depth?
  2. **Q2 (Prefix Tree Hashing):** How are prefix cache blocks indexed using cryptographic or locality-sensitive hashes of token IDs to allow constant-time $O(1)$ cache lookup on incoming requests?
  3. **Q3 (Multi-Tenant Cache Security):** In a multi-tenant LLM platform where different users share the same cluster, how do you enforce strict security boundaries to prevent User A from reading cached KV states generated by User B?

---

### Week 6: Distributed Inference & GPU Hardware Architectures
* **Weekly Focus:** Model parallelism (Tensor, Pipeline, Expert/MoE), GPU hardware architectures (Hopper, Blackwell), containerization, and disaggregation.
* **Weekly Goal:** Deploy multi-GPU LLM inference setups using tensor/pipeline parallelism and understand modern GPU memory hierarchies.
* **3 End-of-Week Master Questions:**
  1. *How does Megatron-LM Tensor Parallelism split the Attention Linear layers ($Q, K, V$ projections vs. Output projection) across GPUs to minimize All-Reduce communication steps?*
  2. *How do Hopper SMs, Tensor Memory Accelerator (TMA), and FP8 Transformer Engines accelerate LLM tensor computations compared to previous GPU generations?*
  3. *Why does Mixture-of-Experts (MoE) routing introduce dynamic load imbalance across GPUs, and how do expert capacity limits mitigate worker starvation?*

#### Day 31 (Mon): Chip Huyen Ch 7 — Deployment Paradigms
* **Topic:** Batch Prediction vs. Real-Time Online Endpoints & Edge vs. Cloud
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Batch vs. Online Trade-offs):** An e-commerce platform needs to show personalized product recommendations. Compare pre-computing recommendations in a nightly batch job (stored in Redis) versus scoring items on-demand via a real-time API. What are the latency, cost, and freshness trade-offs?
  2. **Q2 (Edge Deployment Constraints):** You are deploying an object detection model to an edge device (e.g., security camera with 2GB VRAM). How do power consumption, thermal throttling, and lack of cloud connectivity dictate your model architecture choices?
  3. **Q3 (Hybrid Deployment):** How do you architect a hybrid inference pipeline where a fast, small edge model handles 90% of easy requests and escalates ambiguous, low-confidence requests to a heavy cloud LLM?

#### Day 32 (Tue): Chip Huyen Ch 7 — Scaling Prediction Services
* **Topic:** Horizontal Scaling, Load Balancing, Circuit Breakers, & Fallbacks
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Cascading Failures):** During a traffic surge, your primary LLM inference service hits 100% GPU utilization. Retries from client services double the traffic, causing a cascading failure. How do you implement exponential backoff with jitter and circuit breakers to stabilize the system?
  2. **Q2 (Graceful Degradation):** When an online recommendation service experiences severe latency degradation during peak hours, what dynamic fallback strategies (e.g., dropping heavy re-ranking stages, serving cached popular items) can maintain sub-100ms response times?
  3. **Q3 (Stateless vs. Stateful Endpoints):** Why must prediction service pods in Kubernetes remain completely stateless, and where should session state/history be offloaded?

#### Day 33 (Wed): Chip Huyen Ch 7 — Model Compression in Production
* **Topic:** Knowledge Distillation, Structural Pruning, & Production Trade-offs
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Teacher-Student Distillation):** When training a 1B student transformer from a 70B teacher model using KL-Divergence loss over soft logits, why does soft logit matching transfer dark knowledge that hard label training cannot capture?
  2. **Q2 (Structured vs. Unstructured Pruning):** Why does unstructured zeroing out of weights fail to deliver speedups on standard GPU hardware without specialized sparse tensor cores, whereas structured pruning (removing full attention heads/layers) guarantees immediate speedups?
  3. **Q3 (Quantization Accuracy Degradation):** During post-training quantization of a fine-tuned domain LLM, accuracy drops sharply on specialized medical terminology. How do you identify which specific model layers require mixed-precision retention (e.g., keeping early layers at FP16)?

#### Day 34 (Thu): Day 17 — Model Parallelism
* **Topic:** Tensor, Pipeline, Expert (MoE), & Data Parallelism in LLM Inference (Inference Eng Ch 5.4)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Tensor Parallelism All-Reduce):** In Megatron-LM Tensor Parallelism across 4 GPUs, trace the execution of a ColumnParallel Linear layer followed by a RowParallel Linear layer. Where do the `All-Reduce` communication primitives occur?
  2. **Q2 (Pipeline Parallelism Bubble):** What is the "pipeline bubble" in Pipeline Parallelism (PP), and how do micro-batching schedules (e.g., 1F1B schedule) minimize GPU idle time?
  3. **Q3 (MoE Expert Routing & Capacity):** In a Mixture-of-Experts model (e.g., Mixtral 8x7B), what happens when the top-k router sends 90% of tokens to Expert 1, causing an imbalance? How does Expert Capacity Limit enforce token dropping or secondary routing?

#### Day 35 (Fri): Day 18 — Disaggregation: Prefill/Decode Split
* **Topic:** Prefill/Decode Decoupled Architecture & Chunked Prefills (Inference Eng Ch 5.5)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Chunked Prefills):** How does Chunked Prefill break long prompt sequences into smaller chunks and piggyback them onto Decode iteration batches to prevent inter-token latency spikes on unified workers?
  2. **Q2 (Network Throughput Sizing):** If a Prefill cluster processes 10,000 prompt tokens/sec and generates KV cache tensors at 2MB per token, calculate the required inter-cluster network bandwidth (in Gbps) to transfer these states to the Decode cluster without queuing.
  3. **Q3 (SLA Optimization):** Under what conditions of prompt length ($P$) vs. generation length ($G$) does a disaggregated serving architecture yield the highest cost savings compared to unified serving?

#### Day 36 (Sat): Day 19 & 20 — GPU Architecture & Generations
* **Topic:** SMs, Memory Hierarchy (SRAM, HBM3e), Hopper, Blackwell, & FP4 (Inference Eng Ch 3.1–3.2)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (GPU Architecture Evolution):** Compare NVIDIA Hopper (H100) vs. Blackwell (B200) architectures in terms of NVLink bandwidth, FP4 Tensor Core support, and native Decompression Engines.
  2. **Q2 (Asynchronous Memory Transfers):** How does the Tensor Memory Accelerator (TMA) in Hopper GPUs transfer data directly between global HBM3 memory and Shared Memory (SRAM) asynchronously without consuming SM compute clock cycles?
  3. **Q3 (FP4 Arithmetic Precision):** What numerical challenges arise when executing inference in 4-bit floating point (NVFP4), and how do block scaling factors stabilize attention matrix calculations?

---

### Week 7: Production Observability, Shifts, & Autoscaling
* **Weekly Focus:** Data distribution shifts, ML observability, canary deployments, autoscaling policies, and request routing.
* **Weekly Goal:** Implement real-time drift detection pipelines and build robust autoscaling/load-balancing configurations for production LLM clusters.
* **3 End-of-Week Master Questions:**
  1. *How do you mathematically distinguish between Covariate Shift, Label Shift, and Concept Drift in a deployed production model?*
  2. *Why do traditional CPU-utilization-based autoscaling policies fail for LLM inference workloads, and why must autoscaling trigger on KV-cache memory usage and queue delay?*
  3. *How do zero-downtime blue/green deployment strategies handle schema changes in streaming feature stores without dropping active prediction requests?*

#### Day 37 (Mon): Chip Huyen Ch 8 — Data Distribution Shifts & Drift Detection
* **Topic:** Covariate Shift, Label Shift, Concept Drift, & Statistical Distance Tests
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Drift Classification):** A credit scoring model trained during economic growth experiences a sudden spike in default rates during a recession, even though input applicant demographics remain unchanged. Is this Covariate Shift, Label Shift, or Concept Drift?
  2. **Q2 (Statistical Drift Metrics):** Compare the Kolmogorov-Smirnov (KS) test, Population Stability Index (PSI), and Wasserstein Distance for detecting feature distribution shifts in continuous production features.
  3. **Q3 (Drift Alerting Calibration):** Why does calculating statistical drift on high-cardinality streaming features trigger frequent false-positive alerts, and how do windowing strategies (e.g., sliding 7-day windows) stabilize alerting?

#### Day 38 (Tue): Chip Huyen Ch 8 — Monitoring & Observability in Production
* **Topic:** Operational vs. ML Metrics, Logging Architectures, & Outlier Detection
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Observability Stack):** Design the logging architecture for a low-latency (sub-50ms) prediction API. How do you log inputs, features, and model predictions asynchronously without blocking the main worker response path?
  2. **Q2 (Operational vs. ML Metrics):** Your Prometheus dashboard shows normal CPU/GPU utilization and 200 OK statuses, but customer support reports garbage outputs. What specific ML-level monitoring metrics (e.g., prediction distribution entropy, embedding drift) were missing?
  3. **Q3 (Outlier Detection at Inference):** How can an isolation forest or autoencoder model running alongside an inference service flag out-of-distribution (OOD) inputs in real time?

#### Day 39 (Wed): Chip Huyen Ch 9 — Continual Learning & Testing in Prod
* **Topic:** Shadow Deployment, A/B Testing, Canary Releases, & Retraining Triggers
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Shadow Deployment Mechanics):** How does a shadow deployment work, and how do you analyze shadow model metrics against production baseline metrics without impacting live user traffic?
  2. **Q2 (A/B Test Sample Size Calculation):** You want to test if Model B increases user conversion rate by 0.5% over Model A (baseline conversion = 5%). How do you determine the required sample size and exposure duration to achieve statistical significance ($p < 0.05$)?
  3. **Q3 (Automated Retraining Triggers):** What are the risks of triggering automated model retraining based purely on a scheduled timer versus triggering based on drift threshold breaches or performance drops?

#### Day 40 (Thu): Day 21 & 22 — Multi-GPU Instances, MIG & Containerization
* **Topic:** Multi-Instance GPU (MIG), Docker, NVIDIA Container Toolkit, & NVIDIA NIMs (Inference Eng Ch 3.3, 7.1)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (MIG Partitioning):** How does Multi-Instance GPU (MIG) slice an NVIDIA A100/H100 into isolated hardware instances with dedicated SMs, memory controllers, and crossbars? When is MIG preferable to multi-tenant process sharing (MPS)?
  2. **Q2 (NVIDIA Container Runtime):** How does the NVIDIA Container Toolkit expose host GPU devices, CUDA drivers, and NVML libraries inside Docker containers without mounting full driver binaries?
  3. **Q3 (NVIDIA NIM Architecture):** What advantages do pre-packaged inference microservices (NVIDIA NIMs) offer over custom Docker containers when deploying optimized LLM engines to Kubernetes clusters?

#### Day 41 (Fri): Day 23 — Autoscaling & Cold Starts
* **Topic:** Concurrency Scaling, Dynamic Batching, & Serverless Cold Start Mitigation (Inference Eng Ch 7.2)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Autoscaling Metrics for LLMs):** Why is scaling inference pods based on CPU/RAM utilization ineffective for LLMs? Explain how using custom metrics like `vLLM:num_requests_waiting` and `vLLM:gpu_cache_usage_perc` enables proactive scaling.
  2. **Q2 (Cold Start Bottleneck):** A serverless GPU worker takes 45 seconds to cold start (pulling a 14GB model container and loading weights into VRAM). What architectural techniques (e.g., model weight streaming, warm pool reserves, shared volume mounts) reduce cold start time to under 3 seconds?
  3. **Q3 (Scale-Down Thashing):** How do you configure scale-down stabilization windows and cooldown periods in Kubernetes Horizontal Pod Autoscaler (HPA) to prevent cluster thrashing during fluctuating traffic patterns?

#### Day 42 (Sat): Day 24 — Routing, Load Balancing & Queueing
* **Topic:** Priority Queues, Request Shedding, & KV-Cache-Aware Load Balancers (Inference Eng Ch 7.2.3)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Request Shedding Policies):** When an LLM endpoint reaches maximum queue capacity, what request shedding strategies (e.g., CoDel, drop-tail, dropping requests exceeding deadline SLAs) preserve system stability?
  2. **Q2 (Priority Queueing):** How do you design a multi-tier inference queue that guarantees sub-second latency for paid SLA API tier users while processing free-tier requests in lower-priority batch queues?
  3. **Q3 (Consistent Hashing for Caching):** How does consistent hashing over incoming prompt prefixes allow an ingress load balancer to route requests with matching prompts to the same GPU worker to maximize prefix cache hits?

---

### Week 8: Multi-Cloud MLOps, Cost Optimization, & Capstone System Design
* **Weekly Focus:** Multi-cloud capacity management, zero-downtime deployments, cost estimation, CI/CD pipelines, and end-to-end LLM platform system design.
* **Weekly Goal:** Master the financial and operational engineering of large-scale ML infrastructure and pass a comprehensive capstone system design scenario.
* **3 End-of-Week Master Questions:**
  1. *How do you build a resilient, cost-effective GPU serving cluster using a mix of spot/preemptible instances and reserved fallback nodes across multi-cloud regions?*
  2. *What is the Total Cost of Ownership (TCO) calculation comparing self-hosting an open LLM (Llama 3 70B on 4x H100s) vs. consuming a commercial LLM API (e.g., OpenAI GPT-4o) at 50 million tokens/day?*
  3. *How do you architect an enterprise multi-tenant AI platform that balances strict latency SLAs, cost optimization, automated drift-based retraining, and zero-downtime blue/green rollouts?*

#### Day 43 (Mon): Chip Huyen Ch 10 — MLOps Infrastructure & Tooling
* **Topic:** Experiment Tracking, Artifact Registries, & Pipeline Orchestration DAGs
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Artifact Versioning):** Why must ML artifact registries enforce strict immutability and cryptographic hashing over trained model weights, datasets, and execution environment specifications?
  2. **Q2 (Pipeline Orchestration):** Compare DAG orchestrators (e.g., Airflow, Prefect, Kubeflow Pipelines) for managing ML pipelines. How do you handle conditional branching when an evaluation step fails validation gates?
  3. **Q3 (Metadata Store Architecture):** What schema design is needed in an ML metadata store (e.g., MLflow) to trace a production prediction back through model version, training run, git commit, training data snapshot, and raw source logs?

#### Day 44 (Tue): Chip Huyen Ch 10 & 11 — CI/CD for Machine Learning
* **Topic:** Continuous Integration, Continuous Delivery, & Continuous Testing (CT/CD)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (ML CI/CD Testing Pipeline):** Describe the multi-stage CI/CD pipeline triggered when a data scientist submits a Pull Request modifying a feature extraction script or model architecture code.
  2. **Q2 (Automated Model Quality Gates):** What non-deterministic testing challenges arise when writing automated unit/integration tests for LLM outputs, and how do assertion-based LLM-as-a-judge frameworks enforce quality gates in CI/CD?
  3. **Q3 (Canary Rollout Automation):** How do you configure a progressive canary release via GitOps (Argo Rollouts) that automatically shifts traffic from $1\%$ to $100\%$ over 4 hours while continuously monitoring error rates and latency p99 metrics?

#### Day 45 (Wed): Chip Huyen Ch 11 — Responsible AI & Human-in-the-Loop
* **Topic:** Model Cards, Bias Mitigation, Auditability, & Human Feedback Loops
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Model Card Auditing):** What critical system operational boundaries, failure modes, intended usage contexts, and evaluation demographics must be documented in a production Model Card?
  2. **Q2 (Human-in-the-Loop Architecture):** Design the feedback loop infrastructure for a medical report summarization model where human clinicians correct model outputs. How are these corrections ingested into a fine-tuning dataset without corrupting model behavior?
  3. **Q3 (Algorithmic Bias Audit):** How do you detect and quantify demographic parity or equalized odds disparities in a machine learning model deployed for automated loan approval?

#### Day 46 (Thu): Day 25 — Multi-Cloud Capacity Management
* **Topic:** GPU Spot Instances, Multi-Cloud Failover, & Compute Cost Orchestration (Inference Eng Ch 7.3)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (Spot Instance Preemption Handling):** GPU spot instances offer 60-80% cost savings but can be reclaimed by cloud providers with a 30-second warning. How do you architect a stateless inference cluster that handles sudden node termination without dropping active HTTP client connections?
  2. **Q2 (Multi-Cloud Failover Router):** How do you construct a dynamic DNS / Global Server Load Balancing (GSLB) routing layer that automatically shifts inference traffic between AWS, GCP, Azure, and specialized GPU clouds (e.g., RunPod, Lambda Labs) based on live quota availability and latency?
  3. **Q3 (GPU Cost Estimation Model):** Given GPU hourly rates ($H100 = $3.00/hr$), target concurrency, average prompt length, and generation length, write the equation to estimate monthly cloud compute spend for an enterprise LLM service.

#### Day 47 (Fri): Day 26 — Zero-Downtime Deployment & Cost Estimation
* **Topic:** Blue/Green Deployments, SLA/SLO Definitions, & Total Cost of Ownership (TCO) (Inference Eng Ch 7.4)
* **3 Senior MLE Practice Questions:**
  1. **Q1 (TCO Analysis):** Compare the Total Cost of Ownership (TCO) of self-hosting Llama 3 70B on 4x H100 instances ($12/hr compute + bandwidth + engineering maintenance) vs. calling an external API ($0.60 per 1M tokens). At what daily token volume threshold does self-hosting become cheaper?
  2. **Q2 (Zero-Downtime Model Swapping):** How do you perform a zero-downtime blue/green model update on a single GPU server where loading the new model into VRAM requires 40GB, but the old model currently occupies 40GB of the available 80GB VRAM?
  3. **Q3 (SLA vs. SLO vs. SLI):** Define Service Level Indicators (SLI), Service Level Objectives (SLO), and Service Level Agreements (SLA) for an enterprise LLM inference service. What error budget management strategy triggers a code freeze when SLOs are breached?

#### Day 48 (Sat): Capstone Systems Integration
* **Topic:** Comprehensive System Design: End-to-End Enterprise LLM Platform
* **3 Senior MLE Capstone Practice Questions:**
  1. **Capstone Q1 (Architecture Blueprint):** Design an end-to-end real-time AI Customer Assistant platform for a bank processing 100 million requests/day with an SLA of $p95 < 200ms$. Draw the full architecture from client gateway, stream ingestion, feature store, disaggregated vLLM/TensorRT-LLM serving cluster, speculative decoding, to drift monitoring.
  2. **Capstone Q2 (Failure Recovery & Resilience):** During a high-traffic event, GPU Node 4 experiences an uncorrectable SRAM ECC error and crashes, the primary feature store Redis node experiences network partitioning, and prompt injection attacks double incoming token lengths. Walk through how your system architecture isolates and recovers from each failure mode simultaneously.
  3. **Capstone Q3 (Optimization & Economics):** Your executive team orders a 40% reduction in cloud AI infrastructure costs without degrading user satisfaction or breaching p95 latency SLOs. Detail the exact technical stack changes (quantization, prefix caching, disaggregation, model distillation, spot routing) you will implement to achieve this goal.

---
