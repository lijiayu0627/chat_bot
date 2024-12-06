import argparse
import rpyc
from rpyc.utils.server import ThreadedServer
from build_database import ProductDatabase, FAQDatabase
from rag import RAGSystem


class AnswerGeneration(rpyc.Service):
    """
        An RPyC service class that integrates with the RAG system,
        processing user queries and generating answers.
    """
    def __init__(self):
        """
            Initializes the AnswerGeneration service by setting up the ProductDatabase and ServiceDatabase
            and creating an instance of the RAG system.
        """
        super().__init__()
        self.product_database = ProductDatabase()
        self.product_database.build_database()
        self.faq_database = FAQDatabase()
        self.faq_database.build_database()
        # Create the RAG system instance
        self.rag_system = RAGSystem(self.faq_database, self.product_database, api_key)
        # Tracks if the current query is the first query
        self.initial = True

    def exposed_generate_answer(self, query):
        """
            Exposed method for generating answers to queries.

            Args:
                query (str): The user's query.

            Returns:
                str: The generated answer from the RAG system.
        """
        ans = self.rag_system.ask(query, self.initial)
        self.initial = False
        return ans


if __name__ == '__main__':
    """
        Main function to start the AnswerGeneration service.
        Parses the API key from command-line arguments and initializes the RPyC server.
    """
    # Set up argument parser to retrieve the API key
    parser = argparse.ArgumentParser(description="Answer Generation Server")
    parser.add_argument('-k', '--key', type=str, required=True, help='Your OpenAI API Key')
    args = parser.parse_args()
    api_key = args.key
    # Start the RPyC threaded server
    server = ThreadedServer(AnswerGeneration, port=18862)
    print("AnswerService is running on port 18862...")
    server.start()
