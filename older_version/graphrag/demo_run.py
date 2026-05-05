"""
=============================================================================
DEMO RUN — Full Pipeline with Synthetic Research Papers
=============================================================================
This script:
  1. Generates 3 synthetic research paper PDFs with realistic content
  2. Runs the full GraphRAG pipeline (all 4 layers)
  3. Executes 4 example queries demonstrating all 3 agent types
  4. Prints detailed output at each stage

Run: python pipeline/demo_run.py
=============================================================================
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import fitz  # PyMuPDF — used to CREATE synthetic PDFs

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Synthetic Paper Content
# ---------------------------------------------------------------------------

PAPERS = {
    "attention_is_all_you_need_summary": {
        "title": "Attention Is All You Need: A Summary",
        "sections": {
            "Abstract": """
We present the Transformer, a novel neural network architecture based entirely on attention
mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine
translation tasks show these models to be superior in quality while being more parallelizable
and requiring significantly less time to train. The Transformer achieves 28.4 BLEU on the
WMT 2014 English-to-German translation task, improving over the existing best results,
including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,
our model establishes a new single-model state-of-the-art BLEU score of 41.0, outperforming
all of the previously published single models, at less than 1/4 the training cost of the
previous state-of-the-art model.
            """,
            "Introduction": """
Recurrent neural networks, long short-term memory (LSTM) and gated recurrent neural networks
have been firmly established as state-of-the-art approaches in sequence modeling and transduction
problems such as language modeling and machine translation. Numerous efforts have since continued
to push the boundaries of recurrent language models and encoder-decoder architectures.

Recurrent models typically factor computation along the symbol positions of the input and output
sequences. The Transformer model architecture eschews recurrence and instead relies entirely on
an attention mechanism to draw global dependencies between input and output, allowing for
significantly more parallelization.
            """,
            "Methods": """
The Transformer follows an encoder-decoder structure. The encoder maps an input sequence of
symbol representations to a sequence of continuous representations. Given this, the decoder
then generates an output sequence of symbols one element at a time.

The encoder is composed of a stack of N=6 identical layers. Each layer has two sub-layers:
a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.
We employ a residual connection around each of the two sub-layers, followed by layer normalization.

Multi-head attention allows the model to jointly attend to information from different representation
subspaces at different positions. With a single attention head, averaging inhibits this capability.
The model uses d_model=512 and h=8 parallel attention heads.
            """,
            "Results": """
On the WMT 2014 English-to-German translation task, the big Transformer model outperforms the
best previously reported models including ensembles by more than 2.0 BLEU, establishing a new
state-of-the-art BLEU score of 28.4. The base Transformer model achieves 27.3 BLEU.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0,
outperforming all previously published single models, at less than 1/4 the training cost of the
previous state-of-the-art model.

Training the big model took 3.5 days on 8 P100 GPUs. The base model trained for 12 hours,
representing a significant improvement in training efficiency over LSTM-based models.
            """,
            "Conclusion": """
In this work, we presented the Transformer, the first sequence transduction model based entirely
on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures
with multi-headed self-attention.

For translation tasks, the Transformer can be trained significantly faster than architectures
based on recurrent or convolutional layers. We achieved new state of the art on the WMT 2014
English-to-German and WMT 2014 English-to-French translation tasks.

We plan to extend the Transformer to problems involving input and output modalities other than
text and to investigate local, restricted attention mechanisms to efficiently handle large inputs
and outputs such as images, audio and video.
            """,
        }
    },

    "bert_pretraining": {
        "title": "BERT: Pre-training Deep Bidirectional Transformers",
        "sections": {
            "Abstract": """
We introduce a new language representation model called BERT, which stands for Bidirectional
Encoder Representations from Transformers. Unlike recent language representation models, BERT is
designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning
on both left and right context in all layers. As a result, the pre-trained BERT model can be
fine-tuned with just one additional output layer to create state-of-the-art models for a wide
range of tasks, such as question answering and language inference, without substantial task-specific
architecture modifications.

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on
eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point
absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question
answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1
(5.1 point absolute improvement).
            """,
            "Introduction": """
Language model pre-training has been shown to be effective for improving many natural language
processing tasks. These include sentence-level tasks such as natural language inference and
paraphrasing, which aim to predict the relationships between sentences by analyzing them holistically,
as well as token-level tasks such as named entity recognition and question answering, where models
are required to produce fine-grained output at the token level.

There are two existing strategies for applying pre-trained language representations to downstream
tasks: feature-based and fine-tuning. The feature-based approach, such as ELMo, uses task-specific
architectures that include the pre-trained representations as additional features. The fine-tuning
approach, such as the Generative Pre-trained Transformer (OpenAI GPT), introduces minimal
task-specific parameters, and is trained on the downstream tasks by simply fine-tuning all
pre-trained parameters.
            """,
            "Methods": """
BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original
Transformer implementation. We primarily report results on two model sizes:
BERT_BASE (L=12, H=768, A=12, Total Parameters=110M) and
BERT_LARGE (L=24, H=1024, A=16, Total Parameters=340M).

Pre-training BERT uses two unsupervised tasks. Task 1 is Masked Language Model (MLM):
we simply mask some percentage of the input tokens at random, and then predict those masked tokens.
In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random.

Task 2 is Next Sentence Prediction (NSP): many important downstream tasks such as Question
Answering and Natural Language Inference are based on understanding the relationship between two
sentences, which is not directly captured by language modeling. We pre-train a binarized NSP task
that can be trivially generated from any monolingual corpus.
            """,
            "Results": """
BERT obtains new state-of-the-art results on eleven NLP tasks. Results on the GLUE benchmark:
BERT_LARGE achieves a score of 80.5, which represents a 7.7% absolute improvement over the
previous state of the art. On SQuAD v1.1, BERT_LARGE achieves Test F1 of 93.2 (1.5 point
improvement). On SQuAD v2.0, BERT achieves Test F1 of 83.1 (5.1 point improvement).

Ablation studies show that both pre-training objectives (MLM and NSP) contribute substantially to
BERT's strong performance. Removing NSP hurts performance on QNLI, MNLI and SQuAD 1.1, and
using a left-to-right model (no bidirectionality) causes a significant drop on all tasks.

The model pre-trained on BooksCorpus (800M words) and English Wikipedia (2,500M words) demonstrates
that scale in pre-training data is crucial to downstream task performance.
            """,
        }
    },

    "gpt_language_models": {
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "sections": {
            "Abstract": """
We show that scaling up language models greatly improves task-agnostic, few-shot performance,
sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches.
Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters,
10x more than any previous non-sparse language model, and test its performance in the few-shot
setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks
and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves
strong performance on many NLP datasets, including translation, question-answering, and cloze tasks,
as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling
words, using a novel word in a sentence, or performing 3-digit arithmetic.
            """,
            "Introduction": """
Recent years have featured a trend towards pre-trained language representations in NLP systems,
applied in increasingly flexible and task-agnostic ways. First, single-layer representations were
learned using word vectors and fed to task-specific architectures. Then recurrent networks were used
to produce context-sensitive representations. More recently, Transformers with pre-training have
been used to produce rich, contextual word representations, leading to powerful transfer learning.

A major limitation of these approaches is that while the architecture is generic, the paradigm
still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples.
GPT-3 demonstrates that language models can be meta-learners: they store knowledge in their
parameters during pre-training and can then use that knowledge at inference time given just a few
demonstrations in the context.
            """,
            "Methods": """
GPT-3 uses the same model and architecture as GPT-2, including the modified initialization,
pre-normalization, and reversible tokenization described therein, with the exception that we use
alternating dense and locally banded sparse attention patterns in the layers of the transformer,
similar to the Sparse Transformer.

The model was trained on a blend of datasets: Common Crawl (filtered, 410 billion tokens),
WebText2 (19 billion tokens), Books1 (12 billion tokens), Books2 (55 billion tokens), and
Wikipedia (3 billion tokens). The model has 175B parameters, 96 layers, 96 attention heads,
and a context window of 2048 tokens.

In-context learning is evaluated in three settings: few-shot (10-100 examples in context),
one-shot (exactly one example), and zero-shot (no examples, only task description).
            """,
            "Results": """
GPT-3 achieves strong performance across many benchmarks. On SuperGLUE, GPT-3 achieves 71.8
in the few-shot setting, approaching fine-tuned BERT_LARGE performance of 69.0. On TriviaQA,
GPT-3 achieves 71.2% in the zero-shot setting and 77.5% in the few-shot setting.

On translation tasks, GPT-3 is competitive with supervised models. On WMT'14 English-to-French,
GPT-3 achieves 25.2 BLEU in the zero-shot setting and 28.1 BLEU in the few-shot setting.

GPT-3 demonstrates strong few-shot learning on arithmetic tasks, correctly performing 2-digit
addition 100% of the time, 3-digit addition 80.2%, and 4-digit addition 25.2% — far exceeding
what would be expected from a model not specifically trained for arithmetic.
            """,
            "Conclusion": """
We presented a 175 billion parameter language model, GPT-3, which achieves strong performance on
many NLP benchmarks and tasks in the zero-shot, one-shot, and few-shot settings. GPT-3 demonstrates
that very large language models can be task-agnostic, while still performing well across a diverse
range of tasks. This suggests a path towards even more general language systems.

Key limitations include: lack of interpretability, sample inefficiency relative to humans, potential
for harmful uses, and unclear calibration. Future work should address these limitations while
continuing to scale language models and improve pre-training methodologies.

The relationship between GPT-3, BERT, and the Transformer architecture demonstrates how the
self-attention mechanism introduced by Vaswani et al. has become the dominant paradigm in NLP,
with different design choices (autoregressive vs. masked, scale, pre-training objectives) leading
to models with different strengths.
            """,
        }
    },
}


def create_synthetic_pdfs(output_dir: Path) -> None:
    """Create synthetic research paper PDFs with realistic formatting."""
    output_dir.mkdir(exist_ok=True)

    for filename, paper_data in PAPERS.items():
        pdf_path = output_dir / f"{filename}.pdf"
        if pdf_path.exists():
            print(f"  [Demo] Skipping (already exists): {pdf_path.name}")
            continue

        doc = fitz.open()
        title = paper_data["title"]

        # Title page
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 100), title, fontsize=18, fontname="helv")
        page.insert_text((50, 140), "Synthetic Summary for GraphRAG Demo",
                         fontsize=11, fontname="helv")

        # Section pages
        for section_name, section_text in paper_data["sections"].items():
            page = doc.new_page(width=595, height=842)
            y = 50

            # Section heading
            page.insert_text((50, y), section_name, fontsize=14, fontname="helv")
            y += 30

            # Section body — wrap text
            clean_text = textwrap.dedent(section_text).strip()
            wrapped = textwrap.wrap(clean_text, width=90)

            for line in wrapped:
                if y > 800:
                    page = doc.new_page(width=595, height=842)
                    y = 50
                page.insert_text((50, y), line, fontsize=10, fontname="helv")
                y += 14

        doc.save(str(pdf_path))
        doc.close()
        print(f"  [Demo] Created: {pdf_path.name}")


# ---------------------------------------------------------------------------
# Main Demo
# ---------------------------------------------------------------------------

def run_demo():
    from pipeline.pipeline import GraphRAGPipeline
    from agents.agents import AgentType

    papers_dir = Path(__file__).parent.parent / "papers"

    # Step 0: Create synthetic PDFs
    print("\n" + "="*60)
    print("  DEMO SETUP — Generating Synthetic Research Papers")
    print("="*60)
    create_synthetic_pdfs(papers_dir)

    # Step 1–4: Build pipeline
    pipeline = GraphRAGPipeline(papers_dir=str(papers_dir))
    pipeline.build()

    # Save graph
    graph_path = Path(__file__).parent.parent / "knowledge_graph.json"
    pipeline.save_graph(str(graph_path))

    # Example queries
    demo_queries = [
        {
            "label":       "SUMMARY QUERY",
            "question":    "What are the main contributions of the Transformer and BERT models?",
            "description": "Demonstrates the Summarizer Agent",
        },
        {
            "label":       "TECHNICAL QUERY",
            "question":    "Compare BERT and GPT-3 architectures, pre-training objectives, and benchmark results.",
            "description": "Demonstrates the Technical Writer Agent",
        },
        {
            "label":       "VISUAL QUERY",
            "question":    "Show the relationships between Transformer, BERT, GPT-3, and their datasets and metrics.",
            "description": "Demonstrates the Visualizer Agent",
        },
        {
            "label":       "MULTI-AGENT QUERY",
            "question":    "Summarise and show a diagram of how attention mechanisms are used across the papers.",
            "description": "Demonstrates Orchestrator routing to multiple agents",
        },
    ]

    all_outputs = []

    for q in demo_queries:
        print(f"\n{'#'*60}")
        print(f"  {q['label']}")
        print(f"  {q['description']}")
        print(f"{'#'*60}")

        responses = pipeline.query(q["question"])

        for resp in responses:
            print(resp)
            all_outputs.append({
                "query":      q["question"],
                "agent":      resp.agent_type.value,
                "output":     resp.output,
                "metadata":   resp.metadata,
            })

    # Save all outputs to JSON
    output_path = Path(__file__).parent.parent / "demo_outputs.json"
    import json
    with open(output_path, "w") as f:
        json.dump(all_outputs, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DEMO COMPLETE")
    print(f"  Knowledge graph saved to: knowledge_graph.json")
    print(f"  All agent outputs saved to: demo_outputs.json")
    print("="*60)


if __name__ == "__main__":
    run_demo()