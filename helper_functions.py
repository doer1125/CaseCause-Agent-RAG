import tiktoken
import re 
from langchain_core.documents import Document
import PyPDF2
import pandas as pd
import textwrap

# 尝试导入pylcs，如果失败则设置为None
try:
    import pylcs
except ImportError:
    pylcs = None




def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Calculates the number of tokens in a given string using a specified encoding.

    Args:
        string: The input string to tokenize.
        encoding_name: The name of the encoding to use or model name.

    Returns:
        The number of tokens in the string according to the specified encoding.
    """
    try:
        # Try to get encoding for the model if it's an OpenAI model
        encoding = tiktoken.encoding_for_model(encoding_name)
    except KeyError:
        # For non-OpenAI models like qwen-plus, use cl100k_base encoding (GPT-4 encoding)
        # This is a reasonable default for most modern models
        encoding = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = len(encoding.encode(string))  # Encode the string and count tokens
    return num_tokens


def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents

def replace_double_lines_with_one_line(text):
    """
    Replaces consecutive double newline characters ('\n\n') with a single newline character ('\n').

    Args:
        text: The input text string.

    Returns:
        The text string with double newlines replaced by single newlines.
    """

    cleaned_text = re.sub(r'\n\n', '\n', text)  # Replace double newlines with single newlines
    return cleaned_text


def split_into_chapters(book_path):
    """
    Splits a PDF book into chapters based on chapter title patterns.

    Args:
        book_path (str): The path to the PDF book file.

    Returns:
        list: A list of Document objects, each representing a chapter with its text content and chapter number metadata.
    """

    with open(book_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        documents = pdf_reader.pages  # Get all pages from the PDF

        # Concatenate text from all pages
        text = " ".join([doc.extract_text() for doc in documents])

        # Try multiple chapter patterns
        chapter_patterns = [
            r'(CHAPTER\s+[A-Z]+(?:\s+[A-Z]+)*)',  # CHAPTER ONE
            r'(Chapter\s+[A-Z]+(?:\s+[A-Z]+)*)',  # Chapter One
            r'(Chapter\s+\d+)',  # Chapter 1
            r'(CHAPTER\s+\d+)',  # CHAPTER 1
            r'(\d+\.\s+[A-Z][^\n]+)',  # 1. Introduction
            r'(\d+\s+[A-Z][^\n]+)'  # 1 Introduction
        ]
        
        chapters = []
        for pattern in chapter_patterns:
            chapters = re.split(pattern, text)
            if len(chapters) > 1:
                print(f"使用章节模式: {pattern}")
                break
        
        # 如果所有模式都失败，将整个文档作为一个章节
        if len(chapters) <= 1:
            print("未找到章节，将整个文档作为一个章节")
            chapter_docs = [Document(page_content=text, metadata={"chapter": 1})]
            return chapter_docs

        # Create Document objects with chapter metadata
        chapter_docs = []
        chapter_num = 1
        for i in range(1, len(chapters), 2):
            chapter_title = chapters[i].strip()
            chapter_content = chapters[i + 1].strip() if (i + 1) < len(chapters) else ""
            chapter_text = chapter_title + " " + chapter_content
            doc = Document(page_content=chapter_text, metadata={"chapter": chapter_num, "title": chapter_title})
            chapter_docs.append(doc)
            chapter_num += 1

    return chapter_docs


def extract_book_quotes_as_documents(documents, min_length=50):
    quotes_as_documents = []
    # Correct pattern for quotes longer than min_length characters, including line breaks
    # Match both English quotes (" and ') and Chinese quotes (“ and ”)
    quote_patterns = [
        re.compile(rf'"(.{{{min_length},}}?)"', re.DOTALL),  # English double quotes
        re.compile(rf"'(\w.{{{min_length},}}?)'", re.DOTALL),  # English single quotes with word boundary
        re.compile(rf'“(.{{{min_length},}}?)”', re.DOTALL),  # Chinese quotes
        re.compile(rf'‘(.{{{min_length},}}?)’', re.DOTALL)   # Chinese single quotes
    ]

    for doc in documents:
        content = doc.page_content
        content = content.replace('\n', ' ')
        
        # Try all quote patterns
        for pattern in quote_patterns:
            found_quotes = pattern.findall(content)
            for quote in found_quotes:
                quote_doc = Document(page_content=quote)
                quotes_as_documents.append(quote_doc)
    
    return quotes_as_documents



def escape_quotes(text):
  """Escapes both single and double quotes in a string.

  Args:
    text: The string to escape.

  Returns:
    The string with single and double quotes escaped.
  """
  return text.replace('"', '\\"').replace("'", "\\'")



def text_wrap(text, width=120):
    """
    Wraps the input text to the specified width.

    Args:
        text (str): The input text to wrap.
        width (int): The width at which to wrap the text.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width=width)


def is_similarity_ratio_lower_than_th(large_string, short_string, th):
    """
    Checks if the similarity ratio between two strings is lower than a given threshold.

    Args:
        large_string: The larger string to compare.
        short_string: The shorter string to compare.
        th: The similarity threshold.

    Returns:
        True if the similarity ratio is lower than the threshold, False otherwise.
    """
    
    # 如果pylcs不可用，返回默认值
    if pylcs is None:
        # 使用简单的字符匹配作为默认实现
        common_chars = set(large_string) & set(short_string)
        similarity_ratio = len(common_chars) / len(short_string) if short_string else 0
    else:
        # 使用pylcs计算LCS
        lcs = pylcs.lcs_sequence_length(large_string, short_string)
        similarity_ratio = lcs / len(short_string) if short_string else 0

    # Check if the similarity ratio is lower than the threshold
    return similarity_ratio < th
    

def analyse_metric_results(results_df):
    """
    Analyzes and prints the results of various metrics.

    Args:
        results_df: A pandas DataFrame containing the metric results.
    """

    for metric_name, metric_value in results_df.items():
        print(f"\n**{metric_name.upper()}**")

        # Extract the numerical value from the Series object
        if isinstance(metric_value, pd.Series):
            metric_value = metric_value.values[0]  # Assuming the value is at index 0

        # Print explanation and score for each metric
        if metric_name == "faithfulness":
            print("Measures how well the generated answer is supported by the retrieved documents.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better faithfulness.
        elif metric_name == "answer_relevancy":
            print("Measures how relevant the generated answer is to the question.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better relevance.
        elif metric_name == "context_precision":
            print("Measures the proportion of retrieved documents that are actually relevant.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better precision (avoiding irrelevant documents).
        elif metric_name == "context_relevancy":
            print("Measures how relevant the retrieved documents are to the question.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better relevance of retrieved documents.
        elif metric_name == "context_recall":
            print("Measures the proportion of relevant documents that are successfully retrieved.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better recall (finding all relevant documents).
        elif metric_name == "context_entity_recall":
            print("Measures the proportion of relevant entities mentioned in the question that are also found in the retrieved documents.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better recall of relevant entities.
        elif metric_name == "answer_similarity":
            print("Measures the semantic similarity between the generated answer and the ground truth answer.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates closer semantic meaning between the answers.
        elif metric_name == "answer_correctness":
            print("Measures whether the generated answer is factually correct.")
            print(f"Score: {metric_value:.4f}")
            # Interpretation: Higher score indicates better correctness.



import dill

def save_object(obj, filename):
    """
    Save a Python object to a file using dill.
    
    Args:
    - obj: The Python object to save.
    - filename: The name of the file where the object will be saved.
    """
    with open(filename, 'wb') as file:
        dill.dump(obj, file)
    print(f"Object has been saved to '{filename}'.")

def load_object(filename):
    """
    Load a Python object from a file using dill.
    
    Args:
    - filename: The name of the file from which the object will be loaded.
    
    Returns:
    - The loaded Python object.
    """
    with open(filename, 'rb') as file:
        obj = dill.load(file)
    print(f"Object has been loaded from '{filename}'.")
    return obj

# Example usage:
# save_object(plan_and_execute_app, 'plan_and_execute_app.pkl')
# plan_and_execute_app = load_object('plan_and_execute_app.pkl')

