from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
    ),
    persist_directory='chroma_db',
    collection_name='sample'
)

vector_store.add_documents(documents=docs)
vector_store.get(include=['embeddings', 'documents', 'metadatas'])

"""
{'ids': ['a3dbfc94-b452-4c28-8f09-a801aa987dcb', 'd1622c36-36e4-4b95-b44b-13c9345f9c96', 'a6e2cb6c-72f8-4478-8861-97c9e0666d99', '2e943114-1fd6-4153-975e-04711535ad2c', '4a195098-c817-41ea-acca-9226d406cb15'], 'embeddings': array([[ 0.0099472 ,  0.06914327, -0.05147117, ..., -0.03543335,
         0.01284812,  0.01248282],
       [ 0.00127742,  0.03129857, -0.02375385, ..., -0.00518365,
        -0.03280615,  0.02737713],
       [-0.10265917,  0.02650809,  0.02271502, ..., -0.03359757,
        -0.07984942, -0.01507707],
       [ 0.0212339 , -0.02468553, -0.04494366, ..., -0.10995809,
         0.00572556,  0.09915377],
       [ 0.01873971,  0.04382845, -0.0430425 , ..., -0.07801617,
        -0.07840682, -0.00304186]], shape=(5, 384)), 'documents': ['Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.', "Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.", 'MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.', 'Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.', 'Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.'], 'uris': None, 'included': ['embeddings', 'documents', 'metadatas'], 'data': None, 'metadatas': [{'team': 'Royal Challengers Bangalore'}, {'team': 'Mumbai Indians'}, {'team': 'Chennai Super Kings'}, {'team': 'Mumbai Indians'}, {'team': 'Chennai Super Kings'}]}
"""

# search documents
vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2 # Top k
)

"""
[Document(metadata={'team': 'Mumbai Indians'}, page_content='Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.'),
 Document(metadata={'team': 'Chennai Super Kings'}, page_content='Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.')]
"""

# search with similarity score
vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)

"""
[(Document(metadata={'team': 'Mumbai Indians'}, page_content='Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.'),
  0.35445845127105713),
 (Document(metadata={'team': 'Chennai Super Kings'}, page_content='Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.'),
  0.4085893929004669)]
"""

# meta-data filtering
vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Chennai Super Kings"}
)

"""
[(Document(metadata={'team': 'Chennai Super Kings'}, page_content='MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.'),
  0.6488258242607117),
 (Document(metadata={'team': 'Chennai Super Kings'}, page_content='Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.'),
  0.6566494703292847)]
"""

# update and view documents
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)
vector_store.update_document(document_id='09a39dc6-3ba6-4ea7-927e-fdda591da5e4', document=updated_doc1)
vector_store.get(include=['embeddings','documents', 'metadatas'])

"""
{'ids': ['09a39dc6-3ba6-4ea7-927e-fdda591da5e4',
  '8b561bf2-72ce-4295-8097-27e6f3bcd582',
  'aa800a1a-4b4f-4e58-8fce-ae2279eb385b',
  'eb519603-4222-46d5-a596-11797a4d39b6',
  '1182b187-1e5b-4e2d-a076-0119742940cc'],
 'embeddings': array([[-0.00544442, -0.01907989,  0.00706373, ..., -0.01627786,
         -0.00032134,  0.00724619],
        [-0.00268021, -0.00010323,  0.02815653, ..., -0.01501936,
          0.00590092, -0.01164922],
        [ 0.00092799, -0.00476   ,  0.0124662 , ..., -0.01731381,
          0.00075886,  0.00296567],
        [-0.02714536,  0.00885395,  0.02699314, ..., -0.02592762,
          0.00900617, -0.01999116],
        [-0.01810451,  0.01281202,  0.0347942 , ..., -0.03034012,
         -0.00595078,  0.00521716]]),
 'documents': ["Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
  "Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
  'MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.',
  'Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.',
  'Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.'],
 'uris': None,
 'included': ['embeddings', 'documents', 'metadatas'],
 'data': None,
 'metadatas': [{'team': 'Royal Challengers Bangalore'},
  {'team': 'Mumbai Indians'},
  {'team': 'Chennai Super Kings'},
  {'team': 'Mumbai Indians'},
  {'team': 'Chennai Super Kings'}]}
"""

# delete document and view documents
vector_store.delete(ids=['09a39dc6-3ba6-4ea7-927e-fdda591da5e4'])
vector_store.get(include=['embeddings','documents', 'metadatas'])

"""
{'ids': ['8b561bf2-72ce-4295-8097-27e6f3bcd582',
  'aa800a1a-4b4f-4e58-8fce-ae2279eb385b',
  'eb519603-4222-46d5-a596-11797a4d39b6',
  '1182b187-1e5b-4e2d-a076-0119742940cc'],
 'embeddings': array([[-0.00268021, -0.00010323,  0.02815653, ..., -0.01501936,
          0.00590092, -0.01164922],
        [ 0.00092799, -0.00476   ,  0.0124662 , ..., -0.01731381,
          0.00075886,  0.00296567],
        [-0.02714536,  0.00885395,  0.02699314, ..., -0.02592762,
          0.00900617, -0.01999116],
        [-0.01810451,  0.01281202,  0.0347942 , ..., -0.03034012,
         -0.00595078,  0.00521716]]),
 'documents': ["Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
  'MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.',
  'Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.',
  'Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.'],
 'uris': None,
 'included': ['embeddings', 'documents', 'metadatas'],
 'data': None,
 'metadatas': [{'team': 'Mumbai Indians'},
  {'team': 'Chennai Super Kings'},
  {'team': 'Mumbai Indians'},
  {'team': 'Chennai Super Kings'}]}
"""