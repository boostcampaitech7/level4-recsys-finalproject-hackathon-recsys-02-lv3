CONFIG = {
    'bpr_batch_size': 2048,
    'latent_dim_rec': 64,
    'lightGCN_n_layers': 3,
    'dropout': 0.001,
    'keep_prob': 0.6,
    'A_n_fold': 100,
    'test_u_batch_size': 1000,
    'multicore': 0,
    'lr': 0.001,
    'decay': 1e-4,
    'pretrain': 0,
    'A_split': False,
    'bigdata': False,
    'shuffle': 'shuffle'
}
DB_CONFIG = {
    "dbname": "embedding",
    "user": "postgres",
    "password": "1234",
    "host": "34.64.81.75",
    "port": "5432",
}
MONGODB_USERNAME = "jaypark"
MONGODB_PASSWORD = 'IOWABB9HnR0ew7Bf'
MONGODB_URI = f"mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@cluster0.octqq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"