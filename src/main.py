"""
Main Application Entry Point
Complete resume analysis pipeline with CLI interface.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

# Import all components
from src.utils import get_config, setup_logger, load_config, create_default_config
from src.ingestion import ingest_resumes
from src.vectorstore import ResumeChunker, OllamaEmbeddings, VectorDatabase
from src.retrieval import ResumeRetriever, ResumeReranker
from src.agents import create_llm_client, create_analyzer

# Setup logger
logger = setup_logger(name="resume_rag", level="INFO")


class ResumeRAGPipeline:
    """
    Complete RAG pipeline for resume analysis.
    Orchestrates all components from ingestion to final recommendation.
    """
    
    def __init__(self, config=None):
        """
        Initialize the pipeline.
        
        Args:
            config: Config object (uses default if None)
        """
        self.config = config or get_config()
        
        # Initialize components (lazy loading)
        self._embedder = None
        self._vectordb = None
        self._retriever = None
        self._reranker = None
        self._llm_client = None
        self._analyzer = None
        
        logger.info("ResumeRAGPipeline initialized")
    
    @property
    def embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            logger.info("Initializing embedder...")
            self._embedder = OllamaEmbeddings(
                model=self.config.embedding_model,
                base_url=self.config.ollama_base_url,
                batch_size=self.config.batch_size,
                normalize=self.config.normalize_embeddings,
                cache_embeddings=self.config.cache_embeddings,
                cache_dir=self.config.cache_dir
            )
        return self._embedder
    
    @property
    def vectordb(self):
        """Lazy load vector database."""
        if self._vectordb is None:
            logger.info("Initializing vector database...")
            self._vectordb = VectorDatabase(
                persist_directory=self.config.chroma_db_path,
                embedding_dimension=self.config.embedding_dimension,
                distance_metric="cosine"
            )
        return self._vectordb
    
    @property
    def retriever(self):
        """Lazy load retriever."""
        if self._retriever is None:
            logger.info("Initializing retriever...")
            self._retriever = ResumeRetriever(
                vectordb=self.vectordb,
                embedder=self.embedder
            )
        return self._retriever
    
    @property
    def reranker(self):
        """Lazy load reranker."""
        if self._reranker is None:
            logger.info("Initializing reranker...")
            self._reranker = ResumeReranker(llm_client=self.llm_client)
        return self._reranker
    
    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            logger.info("Initializing LLM client...")
            self._llm_client = create_llm_client(
                model=self.config.llm_model,
                temperature=self.config.temperature
            )
        return self._llm_client
    
    @property
    def analyzer(self):
        """Lazy load analyzer."""
        if self._analyzer is None:
            logger.info("Initializing analyzer...")
            self._analyzer = create_analyzer(llm_client=self.llm_client)
        return self._analyzer
    
    def ingest_and_index(self, force_reindex: bool = False):
        """
        Ingest resumes and index them in vector database.
        
        Args:
            force_reindex: If True, clear existing data and reindex
        """
        logger.info("="*70)
        logger.info("STEP 1: INGESTION AND INDEXING")
        logger.info("="*70)
        
        # Check if database already has data
        stats = self.vectordb.get_statistics()
        if stats['total_documents'] > 0 and not force_reindex:
            logger.info(f"Vector database already contains {stats['total_documents']} documents")
            logger.info("Skipping ingestion (use --force-reindex to rebuild)")
            return
        
        if force_reindex and stats['total_documents'] > 0:
            logger.info("Clearing existing data...")
            self.vectordb.clear_collection()
        
        # Step 1: Ingest resumes
        logger.info("\n1.1. Loading and parsing resumes...")
        resumes = ingest_resumes(
            dataset_path=self.config.dataset_root,
            use_ocr=False,
            extract_tables=True
        )
        logger.info(f"✓ Processed {len(resumes)} resumes")
        
        # Step 2: Chunk resumes
        logger.info("\n1.2. Chunking resumes...")
        chunker = ResumeChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
            max_chunk_size=self.config.max_chunk_size,
            chunking_strategy=self.config.chunking_strategy
        )
        
        all_chunks = chunker.chunk_batch(resumes, add_context_prefix=True)
        flat_chunks = [chunk for resume_chunks in all_chunks for chunk in resume_chunks]
        logger.info(f"✓ Created {len(flat_chunks)} chunks")
        
        # Step 3: Generate embeddings
        logger.info("\n1.3. Generating embeddings...")
        embeddings = self.embedder.embed_chunks(flat_chunks, show_progress=True)
        logger.info(f"✓ Generated {len(embeddings)} embeddings")
        
        # Step 4: Index in vector database
        logger.info("\n1.4. Indexing in vector database...")
        result = self.vectordb.add_documents(chunks=flat_chunks, embeddings=embeddings)
        logger.info(f"✓ Indexed {result['added']} documents")
        logger.info(f"✓ Total documents in database: {result['total_in_collection']}")
        
        logger.info("\n" + "="*70)
        logger.info("INGESTION COMPLETE")
        logger.info("="*70 + "\n")
    
    def search_candidates(self, 
                        job_description: str,
                        role_category: Optional[str] = None,
                        min_experience: float = 0.0,
                        use_reranking: bool = True,
                        override_top_k: Optional[int] = None) -> list:  # ← Add parameter
        """
        Search for candidates matching job description.
        
        Args:
            job_description: Job description/requirements
            role_category: Filter by role (e.g., 'data_scientist')
            min_experience: Minimum years of experience
            use_reranking: Whether to apply reranking
            override_top_k: Override config top_k value (for dynamic adjustment)
        
        Returns:
            List of candidate matches
        """
        logger.info("="*70)
        logger.info("STEP 2: CANDIDATE SEARCH")
        logger.info("="*70)
        
        # Build filters
        filters = {}
        if role_category:
            filters['role_category'] = role_category
        if min_experience > 0:
            filters['years_of_experience'] = {'$gte': min_experience}
        
        # Use override_top_k if provided, otherwise use config value
        top_k_candidates = override_top_k if override_top_k is not None else self.config.top_k_candidates
        
        logger.info(f"\n2.1. Semantic search (retrieving top {top_k_candidates} candidates)...")
        
        # Initial search
        candidates = self.retriever.search_candidates(
            query=job_description,
            top_k_chunks=self.config.top_k_chunks,
            top_k_candidates=top_k_candidates,  # ← Use the dynamic or config value
            filters=filters if filters else None,
            min_similarity=self.config.min_similarity,
            aggregation_method=self.config.aggregation_method
        )
        logger.info(f"✓ Found {len(candidates)} candidates")
        
        # Optional reranking
        if use_reranking and len(candidates) > 0:
            logger.info("\n2.2. Reranking candidates...")
            
            # For reranking, use the config rerank_top_k or all candidates if fewer
            rerank_k = min(self.config.rerank_top_k, len(candidates))
            
            candidates = self.reranker.rerank_candidates(
                query=job_description,
                candidates=candidates,
                method=self.config.rerank_method,
                top_k=rerank_k
            )
            logger.info(f"✓ Reranked to top {len(candidates)}")
        
        logger.info("\n" + "="*70)
        logger.info("SEARCH COMPLETE")
        logger.info("="*70 + "\n")
        
        return candidates
    
    def analyze_candidates(self, candidates: list, job_description: str) -> list:
        """
        Analyze candidates using LLM.
        
        Args:
            candidates: List of CandidateMatch objects
            job_description: Job description
        
        Returns:
            List of CandidateAnalysis objects
        """
        logger.info("="*70)
        logger.info("STEP 3: LLM ANALYSIS")
        logger.info("="*70)
        
        logger.info(f"\n3.1. Analyzing {len(candidates)} candidates with LLM...")
        analyses = self.analyzer.analyze_candidates_batch(
            candidates=candidates,
            job_description=job_description,
            max_candidates=None
        )
        logger.info(f"✓ Analyzed {len(analyses)} candidates")
        
        # Sort by match score
        analyses.sort(key=lambda x: x.match_score, reverse=True)
        
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*70 + "\n")
        
        return analyses
    
    def find_best_k(self, 
                    job_description: str, 
                    top_k: int = 2,
                    role_category: Optional[str] = None,
                    min_experience: float = 0.0) -> list:
        """
        Complete pipeline: Find the best K candidates.
        
        Args:
            job_description: Job description
            top_k: Number of top candidates to return (default: 2)
            role_category: Filter by role
            min_experience: Minimum years of experience
        
        Returns:
            List of top K CandidateAnalysis objects
        """
        logger.info("\n" + "="*70)
        logger.info(f"FINDING BEST {top_k} CANDIDATES")
        logger.info("="*70 + "\n")
        
        # Calculate how many candidates to retrieve initially
        # Get 3x the requested number for better selection after analysis
        search_k = max(top_k * 2, 10)
        logger.info(f"Will retrieve {search_k} candidates initially for analysis")
        
        # Search (pass the dynamic search_k value)
        candidates = self.search_candidates(
            job_description=job_description,
            role_category=role_category,
            min_experience=min_experience,
            use_reranking=True,
            override_top_k=search_k  # ← Pass the dynamic value
        )
        
        if len(candidates) < top_k:
            logger.warning(f"Found only {len(candidates)} candidates (requested {top_k})")
            logger.warning("Consider lowering min_similarity or removing filters")
        
        # Analyze all retrieved candidates
        logger.info(f"Analyzing {len(candidates)} candidates with LLM...")
        analyses = self.analyze_candidates(candidates, job_description)
        
        if len(analyses) < top_k:
            logger.warning(f"Analyzed only {len(analyses)} candidates (requested {top_k})")
        
        # Get top K from analyzed results
        top_candidates = analyses[:top_k]
        
        logger.info("="*70)
        logger.info(f"TOP {len(top_candidates)} CANDIDATES SELECTED")
        logger.info("="*70)
        
        for i, candidate in enumerate(top_candidates, 1):
            logger.info(f"#{i}: {candidate.filename} (Score: {candidate.match_score}/100)")
        
        logger.info("")
        
        return top_candidates
    
    def generate_report(self, 
                       job_description: str,
                       top_candidates: list,
                       output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive hiring report.
        
        Args:
            job_description: Job description
            top_candidates: List of CandidateAnalysis objects
            output_path: Path to save report (optional)
        
        Returns:
            Report text
        """
        logger.info("="*70)
        logger.info("GENERATING HIRING REPORT")
        logger.info("="*70)
        
        report = self.analyzer.generate_hiring_report(
            top_candidates=top_candidates,
            job_description=job_description
        )
        
        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"✓ Report saved to: {output_file}")
        
        return report


def cmd_init(args):
    """Initialize the system (ingest and index resumes)."""
    logger.info("Initializing Resume RAG System...")
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Create pipeline
    pipeline = ResumeRAGPipeline(config)
    
    # Ingest and index
    pipeline.ingest_and_index(force_reindex=args.force)
    
    logger.info("✓ System initialized successfully!")


def cmd_search(args):
    """Search for candidates."""
    logger.info("Searching for candidates...")
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Create pipeline
    pipeline = ResumeRAGPipeline(config)
    
    # Read job description
    if args.job_file:
        with open(args.job_file, 'r', encoding='utf-8') as f:
            job_description = f.read()
    else:
        job_description = args.job_description
    
    # Search
    candidates = pipeline.search_candidates(
        job_description=job_description,
        role_category=args.role,
        min_experience=args.min_experience,
        use_reranking=not args.no_rerank
    )
    
    # Display results
    print("\n" + "="*70)
    print(f"FOUND {len(candidates)} CANDIDATES")
    print("="*70)
    
    for i, candidate in enumerate(candidates, 1):
        print(f"\n{i}. {candidate.filename}")
        print(f"   Email: {candidate.email}")
        print(f"   Experience: {candidate.years_of_experience} years")
        print(f"   Similarity: {candidate.average_similarity:.3f}")
        print(f"   Chunks: {candidate.chunk_count}")
        print(f"   Sections: {', '.join(candidate.section_coverage.keys())}")


def cmd_analyze(args):
    """Analyze and rank candidates."""
    logger.info("Analyzing candidates...")
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Create pipeline
    pipeline = ResumeRAGPipeline(config)
    
    # Read job description
    if args.job_file:
        with open(args.job_file, 'r', encoding='utf-8') as f:
            job_description = f.read()
    else:
        job_description = args.job_description
    
    # Search
    candidates = pipeline.search_candidates(
        job_description=job_description,
        role_category=args.role,
        min_experience=args.min_experience,
        use_reranking=not args.no_rerank
    )
    
    # Analyze
    analyses = pipeline.analyze_candidates(candidates, job_description)
    
    # Display results
    print("\n" + "="*70)
    print(f"ANALYZED {len(analyses)} CANDIDATES")
    print("="*70)
    
    for i, analysis in enumerate(analyses, 1):
        print(f"\n{'#'*70}")
        print(f"RANK #{i}: {analysis.filename}")
        print(f"{'#'*70}")
        print(f"\nMatch Score: {analysis.match_score}/100")
        print(f"Recommendation: {analysis.recommendation.upper()}")
        print(f"Experience: {analysis.years_of_experience} years")
        print(f"Email: {analysis.email}")
        print(f"\nStrengths:")
        for strength in analysis.strengths:
            print(f"  ✓ {strength}")
        print(f"\nConcerns:")
        for weakness in analysis.weaknesses:
            print(f"  • {weakness}")
        print(f"\nAssessment:")
        print(f"  {analysis.overall_assessment}")
    
    # Save results if requested
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        results = [analysis.to_dict() for analysis in analyses]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Results saved to: {output_file}")


def cmd_find_best_k(args):
    """Find the best K candidates."""
    logger.info(f"Finding best {args.top_k} candidates...")
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Create pipeline
    pipeline = ResumeRAGPipeline(config)
    
    # Read job description
    if args.job_file:
        with open(args.job_file, 'r', encoding='utf-8') as f:
            job_description = f.read()
    else:
        job_description = args.job_description
    
    # Find best K
    top_candidates = pipeline.find_best_k(
        job_description=job_description,
        top_k=args.top_k,
        role_category=args.role,
        min_experience=args.min_experience
    )
    
    # Generate report
    report = pipeline.generate_report(
        job_description=job_description,
        top_candidates=top_candidates,
        output_path=args.output
    )
    
    # Display report
    print("\n" + report)
    
    if args.output:
        print(f"\n✓ Report saved to: {args.output}")
    
    # Save JSON results if requested
    if args.json:
        import json
        results = [candidate.to_dict() for candidate in top_candidates]
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ JSON results saved to: {args.json}")


def cmd_config(args):
    """Manage configuration."""
    if args.create:
        create_default_config(args.create)
        print(f"✓ Default configuration created: {args.create}")
    elif args.show:
        config = get_config()
        print("\nCurrent Configuration:")
        print(json.dumps(config.to_dict(), indent=2))
    else:
        print("Use --create or --show")


def cmd_stats(args):
    """Show database statistics."""
    logger.info("Retrieving database statistics...")
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Create pipeline
    pipeline = ResumeRAGPipeline(config)
    
    # Get stats
    stats = pipeline.vectordb.get_statistics()
    
    print("\n" + "="*70)
    print("DATABASE STATISTICS")
    print("="*70)
    print(f"\nCollection: {stats['collection_name']}")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Embedding Dimension: {stats['embedding_dimension']}")
    print(f"Distance Metric: {stats['distance_metric']}")
    print(f"Persist Directory: {stats['persist_directory']}")
    
    if stats.get('sample_role_distribution'):
        print(f"\nRole Distribution (sample):")
        for role, count in stats['sample_role_distribution'].items():
            print(f"  {role}: {count}")
    
    if stats.get('sample_section_distribution'):
        print(f"\nSection Distribution (sample):")
        for section, count in stats['sample_section_distribution'].items():
            print(f"  {section}: {count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Resume RAG System - AI-powered candidate screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Initialize system (ingest and index resumes)
    python -m src.main init --force
    
    # Search for candidates
    python -m src.main search "Senior Data Scientist with Python" --role data_scientist
    
    # Analyze candidates with LLM
    python -m src.main analyze "Senior Data Scientist" --role data_scientist --output results.json
    
    # Find best 2 candidates (default)
    python -m src.main find-best-k "Senior Data Scientist with 5+ years" --role data_scientist
    
    # Find best 5 candidates
    python -m src.main find-best-k "AI Engineer" --top-k 5 --role data_scientist --output report.txt
    
    # Find best 10 candidates with JSON output
    python -m src.main find-best-k --job-file job_desc.txt --top-k 10 --json results.json --output report.txt
    
    # Show statistics
    python -m src.main stats
    
    # Create default config
    python -m src.main config --create config.json
    """
    )
    
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize system (ingest and index)')
    init_parser.add_argument('--force', action='store_true',
                           help='Force reindexing (clear existing data)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for candidates')
    search_parser.add_argument('job_description', type=str, nargs='?',
                              help='Job description (or use --job-file)')
    search_parser.add_argument('--job-file', type=str,
                              help='Path to job description file')
    search_parser.add_argument('--role', type=str,
                              help='Filter by role category')
    search_parser.add_argument('--min-experience', type=float, default=0.0,
                              help='Minimum years of experience')
    search_parser.add_argument('--no-rerank', action='store_true',
                              help='Skip reranking step')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze candidates with LLM')
    analyze_parser.add_argument('job_description', type=str, nargs='?',
                               help='Job description (or use --job-file)')
    analyze_parser.add_argument('--job-file', type=str,
                               help='Path to job description file')
    analyze_parser.add_argument('--role', type=str,
                               help='Filter by role category')
    analyze_parser.add_argument('--min-experience', type=float, default=0.0,
                               help='Minimum years of experience')
    analyze_parser.add_argument('--no-rerank', action='store_true',
                               help='Skip reranking step')
    analyze_parser.add_argument('--output', type=str,
                               help='Save results to JSON file')
    
    # Find best 2 command (main goal!)
    bestk_parser = subparsers.add_parser('find-best-k',
                                     help='Find best K candidates (complete pipeline)')
    bestk_parser.add_argument('job_description', type=str, nargs='?',
                            help='Job description (or use --job-file)')
    bestk_parser.add_argument('--job-file', type=str,
                            help='Path to job description file')
    bestk_parser.add_argument('--top-k', type=int, default=2,
                            help='Number of top candidates to return (default: 2)')
    bestk_parser.add_argument('--role', type=str,
                            help='Filter by role category')
    bestk_parser.add_argument('--min-experience', type=float, default=0.0,
                            help='Minimum years of experience')
    bestk_parser.add_argument('--output', type=str,
                            help='Save report to file')
    bestk_parser.add_argument('--json', type=str,
                            help='Save results to JSON file')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('--create', type=str,
                              help='Create default config file')
    config_parser.add_argument('--show', action='store_true',
                              help='Show current configuration')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(name="resume_rag", level=args.log_level)
    
    # Route to appropriate command
    if args.command == 'init':
        cmd_init(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'find-best-k':
        cmd_find_best_k(args)
    elif args.command == 'config':
        cmd_config(args)
    elif args.command == 'stats':
        cmd_stats(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()