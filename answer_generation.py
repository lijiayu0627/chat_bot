import argparse
import rpyc
from rpyc.utils.server import ThreadedServer
from build_database import ProductDatabase, ServiceDatabase
from rag import RAGSystem


class AnswerGeneration(rpyc.Service):
    def __init__(self):
        super().__init__()
        self.product_database = ProductDatabase()
        self.product_database.build_database()
        self.service_database = ServiceDatabase()
        self.service_database.build_database()
        self.rag_system = RAGSystem(self.service_database, self.product_database, api_key)
        self.initial = True  # Tracks if it's the first query

    def exposed_generate_answer(self, query):
        ans = self.rag_system.ask(query, self.initial)
        self.initial = False
        return ans


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Task Dispatcher")
    parser.add_argument('-k', '--key', type=str, required=True, help='Your OpenAI API Key')
    args = parser.parse_args()
    api_key = args.key

    server = ThreadedServer(AnswerGeneration, port=18862)
    print("AnswerService is running on port 18862...")
    server.start()
