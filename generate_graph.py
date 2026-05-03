#########################################################
### Soubor pro vygenerování grafu z výsledků evaluace ###
#########################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = [
    {"model": "BAAI/bge-m3", "score": 0.6671362295440388, "recall@10": 0.7476823861346231, "recall@20": 0.800080612656187, "recall@30": 0.8246674727932285, "mrr": 0.5463169946581623},
    {"model": "intfloat/multilingual-e5-base", "score": 0.6286391631993755, "recall@10": 0.7138250705360741, "recall@20": 0.7698508665860541, "recall@30": 0.8012898024989923, "mrr": 0.5008603021943276},
    {"model": "intfloat/multilingual-e5-small", "score": 0.6217482368718892, "recall@10": 0.7045546150745667, "recall@20": 0.7654171704957679, "recall@30": 0.7976622329705764, "mrr": 0.49753866956787285},
    {"model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "score": 0.29683824542325443, "recall@10": 0.3438129786376461, "recall@20": 0.3913744457879887, "recall@30": 0.41676743248690046, "mrr": 0.2263761456016669},
    {"model": "sentence-transformers/all-MiniLM-L12-v2", "score": 0.18082496132979387, "recall@10": 0.2067714631197098, "recall@20": 0.24627166465135025, "recall@30": 0.26118500604594924, "mrr": 0.14190520864491996},
    {"model": "ufal/robeczech-base", "score": 0.1528649873466291, "recall@10": 0.18581217251108423, "recall@20": 0.2293430068520758, "recall@30": 0.264006449012495, "mrr": 0.10344420959994638},
    {"model": "sentence-transformers/LaBSE", "score": 0.3550779198865144, "recall@10": 0.4252317613865377, "recall@20": 0.5054413542926239, "recall@30": 0.5485691253526803, "mrr": 0.24984715763647947},
    {"model": "Snowflake/snowflake-arctic-embed-m", "score": 0.21446227427968803, "recall@10": 0.2547359935509875, "recall@20": 0.2946392583635631, "recall@30": 0.32083837162434503, "mrr": 0.1540516953727388},
    {"model": "sentence-transformers/distiluse-base-multilingual-cased-v2", "score": 0.2746542982216745, "recall@10": 0.3164046755340588, "recall@20": 0.36477226924627165, "recall@30": 0.39862958484482064, "mrr": 0.21202873225309793},
    {"model": "sentence-transformers/stsb-xlm-r-multilingual", "score": 0.20353853309407055, "recall@10": 0.24103184199919386, "recall@20": 0.2889963724304716, "recall@30": 0.32285368802902054, "mrr": 0.14729856973638555},
    {"model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "score": 0.30633230624874036, "recall@10": 0.35106811769447804, "recall@20": 0.40669085046352277, "recall@30": 0.4349052801289803, "mrr": 0.23922858908013384},
    {"model": "sentence-transformers/multi-qa-mpnet-base-dot-v1", "score": 0.23109134006503146, "recall@10": 0.27327690447400244, "recall@20": 0.32406287787182586, "recall@30": 0.3550987505038291, "mrr": 0.16781299345157494},
    {"model": "ibm-granite/granite-embedding-107m-multilingual", "score": 0.4780771782193468, "recall@10": 0.5513905683192262, "recall@20": 0.6138653768641676, "recall@30": 0.6465135026199114, "mrr": 0.3681070930695278},
    {"model": "ibm-granite/granite-embedding-278m-multilingual", "score": 0.5191821793740465, "recall@10": 0.5941152760983475, "recall@20": 0.6505441354292624, "recall@30": 0.6831922611850061, "mrr": 0.40678253428759525},
    {"model": "Seznam/simcse-dist-mpnet-paracrawl-cs-en", "score": 0.3256834635117922, "recall@10": 0.38774687625957277, "recall@20": 0.4534461910519952, "recall@30": 0.4840790004030633, "mrr": 0.23258834439012135},
    {"model": "Seznam/retromae-small-cs", "score": 0.3017974010716342, "recall@10": 0.3587263200322451, "recall@20": 0.4328899637243047, "recall@30": 0.47440548166062074, "mrr": 0.21640402263071787},
    {"model": "Seznam/dist-mpnet-czeng-cs-en", "score": 0.15645942071523466, "recall@10": 0.18782748891575976, "recall@20": 0.22732769044740025, "recall@30": 0.2599758162031439, "mrr": 0.10940731841444705},
    {"model": "intfloat/multilingual-e5-large", "score": 0.6418733073895977, "recall@10": 0.7295445384925433, "recall@20": 0.7908101571946796, "recall@30": 0.8166062071745264, "mrr": 0.5103664607351793},
    {"model": "Snowflake/snowflake-arctic-embed-l-v2.0", "score": 0.6455102922865161, "recall@10": 0.7303506650544136, "recall@20": 0.7843611446997178, "recall@30": 0.8129786376461104, "mrr": 0.51824973313467}
]

df = pd.DataFrame(data)

df = df.sort_values('score', ascending=False).reset_index(drop=True)

def shorten_name(name):
    name = name.replace("sentence-transformers/", "s-t/")
    name = name.replace("ibm-granite/", "ibm/")
    return name

df['short_name'] = df['model'].apply(shorten_name)

metrics_to_plot = ['recall@10', 'mrr']
n_metrics = len(metrics_to_plot)
bar_height = 0.2
group_gap = 0.1

y_pos = np.arange(len(df))

fig, ax = plt.subplots(figsize=(10, 5.5))

for i, metric in enumerate(metrics_to_plot):
    position = y_pos + (i - n_metrics / 2) * bar_height + bar_height / 2
    ax.barh(position, df[metric], height=bar_height, label=metric.upper())

ax.set_yticks(y_pos)
ax.set_yticklabels(df['short_name'], fontsize=12)
ax.invert_yaxis()
ax.set_xlabel('Hodnota metriky', fontsize=12)
ax.legend(title='Metriky', fontsize=10, loc='lower right')
ax.grid(axis='x', linestyle='--', alpha=0.6)
ax.set_xlim(0, 0.8)

plt.tight_layout()
plt.show()

print("Graf byl úspěšně vygenerován")