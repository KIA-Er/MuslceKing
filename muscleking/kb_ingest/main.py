"""kb_ingest的命令行入口"""

from dotenv import load_dotenv

load_dotenv()

from muscleking.kb_ingest.cli import main

if __name__ == "__main__":
    main()
