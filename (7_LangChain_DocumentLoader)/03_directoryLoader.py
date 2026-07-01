# fetch all pdf from the directory/folder and load them using PyPDFLoader

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# load and lazy load the documents
docs = loader.lazy_load()

for document in docs:
    print(document.metadata)