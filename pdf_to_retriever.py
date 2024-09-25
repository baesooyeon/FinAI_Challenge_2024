from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import fitz  # PyMuPDF
import tabula
import re

def pre_energy(x):
  sep = "\n\n\n"
  x = x.replace('\n□',sep)
  x = x.replace("ㆍ", sep)
  x = x.replace("○", sep)
  x = x.replace("-", sep)
  # x = re.sub("\d\)",sep, x)
  x = re.sub("\n①|\n②|\n③|\n④|\n⑤", sep, x)
  x = re.sub("^가.|^나.|^다.|^라.", sep, x)

  return x

def pdf_to_chunk(pdf_path, pdfs_opt, tokenizer) :
    # 문서별 기본 설정값
    pdf_nm = pdf_path.split('/')[-1].split('.pdf')[0]
    pdf_opt = pdfs_opt[pdf_nm]
    
    print(f'Processing {pdf_nm}.pdf...')
    
    # 1. 텍스트 및 표 추출
    ## 수연담당님 문서
    if pdf_nm in ['중소벤처기업부_혁신창업사업화자금(융자)', '「FIS 이슈&포커스」 22-2호 《재정성과관리제도》'] :
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        combined_content = [x.page_content for x in pages]
        combined_content = '\n'.join(combined_content)
        
        ## 텍스트 전처리
        if pdf_nm == '중소벤처기업부_혁신창업사업화자금(융자)':
            combined_content = re.sub("\n\d\.", "\n\n\n" , combined_content)
            
        elif pdf_nm == '보건복지부_부모급여(영아수당) 지원':
            combined_content = re.sub("\n\d\.", "\n\n\n" , combined_content)
            
        elif pdf_nm == '「FIS 이슈&포커스」 22-2호 《재정성과관리제도》':
            combined_content = combined_content.replace("\x07", "")
            combined_content = combined_content.replace("\nISSUE", "\n")
            combined_content = combined_content.replace("\nFOCUS", "\n")
            combined_content = combined_content.replace("\n-","\n\n")
            combined_content = combined_content.replace("\n‣", "\n\n")
            combined_content = combined_content.replace("   ", "\n\n")

    elif pdf_nm ==  '보건복지부_부모급여(영아수당) 지원':
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        document = [re.sub("\n\d\.", "~~" ,page.page_content) for page in pages]
        document = [Document(page_content = page.replace("~~","\n\n\n"), meta_data ={"source":pdf_nm+'.pdf'})  for page in document]
        print(document)

    elif pdf_nm == "산업통상자원부_에너지바우처":
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        document = [Document(page_content = pre_energy(page.page_content), meta_data = {"source" : pdf_nm+".pdf"}) for page in pages]


        
    else :
        doc = fitz.open(pdf_path)
        all_content = []
        
        for page_num in range(len(doc)) :
            page = doc.load_page(page_num)
            
            # 텍스트 추출
            text = page.get_text('text')
            if text :
                all_content.append(('text', text))
            
            if pdf_opt['table_opt'] == True:
                tables = tabula.read_pdf(pdf_path, pages=page_num+1, multiple_tables=True, area=None)
                for table in tables:
                    table_text = table.to_string(index=False)
                    all_content.append(('table', table_text))
        
        # 텍스트와 표 데이터를 결합하여 하나의 Document 객체로 생성
        combined_content = ""
        
        for content_type, content in all_content:
            if content_type == 'text':
                combined_content += content + "\n"
                
            elif content_type == 'table':
                combined_content += "\n[TABLE]\n" + content + "\n[TABLE]\n"

    if pdf_nm != "보건복지부_부모급여(영아수당) 지원" and pdf_nm !="산업통상자원부_에너지바우처":               
        metadata = {"source":pdf_nm+'.pdf'}
        document = Document(page_content=combined_content, metadata=metadata)



    # 문서 분할
    def token_len(text) :
        tokens = tokenizer.encode(text)
        return len(tokens)
    
    if pdf_opt['length_function'] == 'token_len':
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=pdf_opt['chunk_size'],
        chunk_overlap = pdf_opt['chunk_overlap'],
        separators = pdf_opt['seperators'],
        length_function = token_len,
    )
        
    else :
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=pdf_opt['chunk_size'],
        chunk_overlap = pdf_opt['chunk_overlap'],
        separators = pdf_opt['seperators'],
        length_function = len,
    )
    if pdf_nm != "보건복지부_부모급여(영아수당) 지원" and pdf_nm != "산업통상자원부_에너지바우처":  
        chunks = text_splitter.split_text(document.page_content)
        chunk_documents = [Document(page_content=chunk, metadata=document.metadata) for chunk in chunks]

    elif pdf_nm == "보건복지부_부모급여(영아수당) 지원":
        chunk_documents = []
        for idx in range(len(document)):
            chunks = text_splitter.split_text(document[idx].page_content)
            for j in chunks:
                chunk_documents.append(Document(page_content = j, metadata = {"source":pdf_nm+'.pdf'}))

    elif pdf_nm == "산업통상자원부_에너지바우처":
        chunk_documents = []
        for idx in range(len(document)):
            chunks = text_splitter.split_text(document[idx].page_content)
            for j in chunks:
                chunk_documents.append(Document(page_content = j, metadata = {"source":pdf_nm+'.pdf'}))
        print(chunk_documents)
    return chunk_documents

def chunk_to_retriever(chunks, pdf_path, pdfs_opt, embeddings) :
    # 문서별 기본 설정값
    pdf_nm = pdf_path.split('/')[-1].split('.pdf')[0]
    pdf_opt = pdfs_opt[pdf_nm]
    
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_type=pdf_opt['search_type'],
                                search_kwargs=pdf_opt['search_kwargs'])
    
    if pdf_opt['ensemble'] == True :
        print('ensemble retriever 생성')
        bm25_retriever = BM25Retriever.from_documents(
            chunks
        )
        if pdf_nm != "산업통상자원부_에너지바우처":
            bm25_retriever.k = pdf_opt['search_kwargs']['k']
        else:
            bm25_retriever.k = 4
        
        if pdf_nm != "산업통상자원부_에너지바우처":
            retriever = EnsembleRetriever(
                retrievers = [bm25_retriever, retriever], weights = [0.5, 0.5]
            )

        else:
            retriever = EnsembleRetriever(
                retrievers = [bm25_retriever, retriever], weights = [0.6,0.4]
            )

            
    
    if pdf_opt['rerank'] == True :
        print('rerank retriever 생성')
        rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        
        # 상위 3개 문서 선택
        if pdf_nm != "산업통상자원부_에너지바우처":
            compressor = CrossEncoderReranker(model=rerank_model, top_n=4)
        else:
            compressor = CrossEncoderReranker(model=rerank_model, top_n=3)
        
        
        # 문서 압축 검색기 초기화
        retriever = ContextualCompressionRetriever(
            base_compressor = compressor, base_retriever = retriever
        )
        
    return retriever
    
    
        
        
    
    
    