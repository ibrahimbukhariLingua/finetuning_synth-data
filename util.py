import os, json, random, re
from itertools import groupby
from datasets import Dataset
from typing import List, Tuple, Dict
import hashlib, faiss, nltk
import numpy as np
from sentence_transformers import SentenceTransformer


# ========================= RAG SIMULATION FOR SYNTHETIC DATA ==================== #
class RAGFormatter:
    def __init__(self, directory: str):
        # ---------------------------------------- Initialization ------------------------------------
        print("Initializing RAGFormatter...")
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs("data/index", exist_ok=True)

        print(f"Loading model from 'all-MiniLM-L6-v2'...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.texts = self._load_texts() 
        self.index_path = self._get_index_path()

        if os.path.exists(self.index_path):
            print(f"Loading existing index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
        else:
            print(f"Creating new index at {self.index_path}")
            self.embeddings = self._embed_texts()
            self.index = self._build_index()
            faiss.write_index(self.index, self.index_path)

    def _get_index_path(self) -> str:
        dir_hash = hashlib.md5(self.directory.encode()).hexdigest()
        return os.path.join("data/index", f"index_{dir_hash}.faiss")

    # ---------------------------------------- File Loading & Embedding ------------------------------------

    def _load_texts(self) -> List[str]:
        files = sorted([f for f in os.listdir(self.directory) if f.endswith('.txt')])
        return [open(os.path.join(self.directory, f), encoding='utf-8').read() for f in files]

    def _embed_texts(self) -> np.ndarray:
        return np.array(self.model.encode(self.texts, show_progress_bar=True, convert_to_numpy=True))

    def _build_index(self):
        index = faiss.IndexFlatL2(self.embeddings.shape[1])
        index.add(self.embeddings)
        return index

    # ---------------------------------------- Retrieval ------------------------------------

    def _retrieve_top_k(self, question: str, gold_passages: List[str], k: int) -> List[str]:
        cleaned_gold_passages = [
            re.sub(r'\[S\d+\]\s*', '', gp).strip()
            for gp in gold_passages
        ]

        q_vec = self.model.encode([question], convert_to_numpy=True)
        _, indices = self.index.search(q_vec, len(self.texts))

        retrieved = []
        for i in indices[0]:
            candidate = self.texts[i]
            if all(cleaned not in candidate for cleaned in cleaned_gold_passages):
                retrieved.append(candidate)
            if len(retrieved) == k:
                break
        return retrieved

    # ---------------------------------------- Formatting ------------------------------------

    def _renumber_and_format_gold_passage(self, passage: str, passage_number: int) -> Tuple[str, Dict[str, str]]:
        old_tags = re.findall(r'\[S\d+\]', passage)
        new_passage = passage
        tag_map = {}
        for i, old_tag in enumerate(old_tags):
            new_tag = f"|S{i+1}]"
            tag_map[old_tag] = f"[P{passage_number}{new_tag}"
            new_passage = new_passage.replace(old_tag, f"[P{passage_number}{new_tag}", 1)
        return new_passage, tag_map

    def _format_regular_passage(self, passage: str, passage_number: int) -> str:
        sentences = nltk.sent_tokenize(passage)
        return " ".join([f"[P{passage_number}|S{s+1}] {sent}" for s, sent in enumerate(sentences)])

    # ---------------------------------------- Citation Handling ------------------------------------

    def _update_gold_text(self, text: str, marker_map: Dict[str, str]) -> str:
        def repl(match):
            tags = re.findall(r'\[S\d+\]', match.group(0))
            return ''.join([marker_map.get(tag, tag) for tag in tags])
        return re.sub(r'((\[S\d+\]){1,})', repl, text)

    # ---------------------------------------- Public API ------------------------------------

    def format_for_question(
        self,
        question: str,
        gold_passages: List[str],
        gold_answer: str,
        gold_thinking: str,
        top_k: int = 3
    ) -> Tuple[List[str], str, str]:
        retrieved_texts = self._retrieve_top_k(question, gold_passages, k=top_k)

        insert_positions = sorted(random.sample(range(len(retrieved_texts) + len(gold_passages)), len(gold_passages)))

        passages_with_gold = []
        gold_passage_numbers = []
        gold_index = 0
        text_index = 0

        full_marker_map = {}

        for i in range(len(retrieved_texts) + len(gold_passages)):
            if i in insert_positions:
                passage_number = i + 1
                gold_passage = gold_passages[gold_index]
                formatted_gold, marker_map = self._renumber_and_format_gold_passage(gold_passage, passage_number)
                passages_with_gold.append(formatted_gold)
                gold_passage_numbers.append(passage_number)
                full_marker_map.update(marker_map)
                gold_index += 1
            else:
                passage_number = i + 1
                formatted = self._format_regular_passage(retrieved_texts[text_index], passage_number)
                passages_with_gold.append(formatted)
                text_index += 1

        updated_gold_answer = self._update_gold_text(gold_answer, full_marker_map)
        updated_gold_thinking = self._update_gold_text(gold_thinking, full_marker_map)

        return passages_with_gold, updated_gold_answer, updated_gold_thinking



# ========================== CLASS CONVERTING DATA TO HF ========================== #
class Data_to_hf():

    def __init__(self, directory_path:str, num_of_samples:int, add_think:bool, add_cite:bool,  rag_format_dir:str = "/data/home/syed.bukhari/finetuning_synth_data/data/rag_format/en-wikipedia-finance.txt"):
        self.directory_path = directory_path
        self.num_of_samples = num_of_samples
        self.add_think = add_think
        self.add_cite = add_cite
        
        # Initialize RAG formatter: simulates a RAG pipeline to add chunks to the prompt template
        self.formatter = RAGFormatter(directory=rag_format_dir)

    # -------- Helper Functions -------- #
    
    def transform_input(self, text, ):
        
        def compress_citations(citations):
            matches = re.findall(r'\[P(\d+)\|S(\d+)\]', citations)
            passage_dict = {}
            for p, s in matches:
                p = int(p)
                s = int(s)
                passage_dict.setdefault(p, []).append(s)

            compressed = []
            for p in sorted(passage_dict.keys()):
                sentences = sorted(passage_dict[p])
                for k, g in groupby(enumerate(sentences), lambda ix: ix[0] - ix[1]):
                    group = list(map(lambda x: x[1], g))
                    if len(group) > 1:
                        compressed.append(f"[P{p}|S{group[0]}-S{group[-1]}]")
                    else:
                        compressed.append(f"[P{p}|S{group[0]}]")
            return ''.join(compressed)

        def extract_statements(text):
            pattern = re.compile(r'<statement>(.*?)<cite>(.*?)</cite></statement>', re.DOTALL)
            return pattern.findall(text)

        def preprocess_statements(statements):
            processed = []
            for content, citation in statements:
                content = content.strip()
                if not citation.strip():
                    processed.append((content, None))
                else:
                    compressed = compress_citations(citation)
                    processed.append((content, compressed))
            return processed

        def merge_statements(processed):
            merged_statements = []
            i = 0
            while i < len(processed):
                content, citation = processed[i]
                j = i + 1
                while j < len(processed) and processed[j][1] == citation:
                    content += " " + processed[j][0]
                    j += 1
                if citation and self.add_cite==True:
                    merged_statements.append(f"{content}<cite>{citation}</cite>")
                else:
                    merged_statements.append(content)
                i = j
            return merged_statements


        # === Pipeline ===

        # Step 1: Extract statements
        statements = extract_statements(text)

        # Step 2: Preprocess statements (can comment out to skip compression)
        statements = preprocess_statements(statements)

        # Step 3: Merge consecutive statements with same citations (optional)
        output = merge_statements(statements)

        return ''.join(output)

    def preprocess_example(self, example: dict) -> dict:
        """
        Processes a single example using a RAGFormatter.

        Args:
            example (dict): A dictionary with at least 'chunk', 'question', and 'answer'.
                            Optionally includes 'thinking' or 'reasoning'.
            formatter (RAGFormatter): An initialized RAGFormatter object.

        Returns:
            dict: A dictionary with formatted passages, updated answer, and updated thinking.
        """
        # Step 1: Ensure chunk_list is available
        gold_passages = example['chunk_list']

        # Step 2: Extract gold thinking
        gold_thinking = example.get('reasoning') or example.get('thinking', '')

        # Step 3: Format using RAGFormatter
        formatted_passages, updated_answer, updated_thinking = self.formatter.format_for_question(
            question=example['question'],
            gold_passages=gold_passages,
            gold_answer=example['answer'],
            gold_thinking=gold_thinking
        )

        # Step 4: Return processed example
        updated_answer = self.transform_input(updated_answer) # applying transformations on answer
        return {
            'chunk': formatted_passages,
            'answer': updated_answer,
            'thinking': updated_thinking,
            'question': example['question']
        }

    def apply_template(self, example):
        # ---------------------------------------- System Prompt ----------------------------------------
        system_prompt = (
            "You are a question answering system which will be provided with a Source passage. "
            "Using the Source passage you will answer the user's question. "
            "You will cite sources which you use when answering the question."
        )

        # ---------------------------------------- Format Source Passages ----------------------------------------
        formatted_passages = "\n".join(
            f"<Passage{i+1}> {p} </Passage{i+1}>" for i, p in enumerate(example['chunk'])
        )

        # ---------------------------------------- User Prompt ----------------------------------------
        prompt = (
            f"Source Passages:\n{formatted_passages}\n\n"
            f"User Question:\n{example['question']}"
        )

        # ---------------------------------------- Assistant Completion ----------------------------------------
        if self.add_think == True:
            completion = f"<think>{example['thinking']}</think>\n{example['answer']}"
        else:
            completion = example['answer']

        # ---------------------------------------- Message Format ----------------------------------------
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': completion}
        ]

        return {'messages': messages}

    # -------- Main Function ----------- #

    def run(self) -> Dataset:
        all_qa_pairs = []
        
        # Function to extract QA pairs from a JSON file
        def extract_qa_pairs(json_file_path:str):
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data.get("qa_pairs")
        
        # Step 1: Extract all JSON file paths in the directory  
        stop = False
        for root, _, files in os.walk(self.directory_path):
            
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    
                    # Step 2: Extract QA pairs from the file
                    qa_pairs = extract_qa_pairs(file_path)
                    
                    # Step 3: Extend the list with extracted QA pairs
                    all_qa_pairs.extend(qa_pairs)
                    
                    # Step 4: Stop if we have enough samples
                    if len(all_qa_pairs) >= self.num_of_samples:
                        all_qa_pairs = all_qa_pairs[:self.num_of_samples]
                        stop = True
                        break
                
            if stop:
                break
        
        # Step 5: Convert to HuggingFace Datasets format
        hf_dataset = Dataset.from_list(all_qa_pairs)
        hf_dataset = hf_dataset.map(lambda x: self.preprocess_example(x), remove_columns=hf_dataset.column_names)
        hf_dataset = hf_dataset.map(self.apply_template, remove_columns=hf_dataset.column_names)
        
        return hf_dataset
