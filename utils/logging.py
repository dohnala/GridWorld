import logging.config

config = {
        'version': 1,
        'formatters': {
            'detailed': {
                'class': 'logging.Formatter',
                'format': '%(asctime)s %(levelname)-8s %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
                'level': 'INFO',
                'formatter': 'detailed',
            },
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        },
    }

logging.config.dictConfig(config)

logger = logging.getLogger("root")
