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
