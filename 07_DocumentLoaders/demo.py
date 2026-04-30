####################################################################################
# Resource ==> https://docs.langchain.com/oss/python/integrations/document_loaders #
####################################################################################

from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader, WebBaseLoader, CSVLoader

# loader = TextLoader(
#     '../07_DocumentLoaders/documents/cricket.txt', 
#     encoding='utf-8'
# )

# loader = PyPDFLoader(
#     '../07_DocumentLoaders/documents/dl-curriculum.pdf'
# )

# loader = DirectoryLoader(
#     path='../07_DocumentLoaders/documents',
#     glob='**/*.pdf',
#     loader_cls=PyPDFLoader
# )

# url = 'https://en.wikipedia.org/wiki/LangChain'
# loader = WebBaseLoader(url)

loader = CSVLoader(
    '../07_DocumentLoaders/documents/Social_Network_Ads.csv'
)

# docs = loader.lazy_load() # For Large Docs (DirectoryLoader)
docs = loader.load()
print(f"{docs}\n")
print(f"Length: {len(docs)}\n")