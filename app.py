from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import json
import uuid
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
import google.generativeai as genai

from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch


from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



DATABASE_URL = "postgresql://postgres:your_new_password@localhost/bijaya"

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn

# Load API key securely
os.environ["GOOGLE_API_KEY"] = "AIzaSyBMAvkffzHd5VT2nNCfmU3uh432cXtBp9A"

@dataclass
class QuizQuestion:
    question: str
    options: List[str]
    correct_answer: int
    explanation: str
    difficulty: str
    topic: str
    chapter: str

@dataclass
class QuizSession:
    session_id: str
    questions: List[QuizQuestion]
    subject: str
    chapter: str
    current_question: int = 0
    score: int = 0
    completed: bool = False
    answers: List[int] = None
    start_time: str = None
    
    def __post_init__(self):
        if self.answers is None:
            self.answers = []
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()

class DeerWalkRAGQuizGenerator:
    def __init__(self):
        """Initialize the RAG Quiz Generator with Google Gemini"""
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.8
        )
        self.vectorstore = None
        self.quiz_template = self._create_quiz_template()
        self.context_template = self._create_context_template()
        self.chapter_structure = self._create_chapter_structure()
        
    def _create_chapter_structure(self) -> Dict[str, List[str]]:
        """Define chapter structure for each subject"""
        return {
            "Programming in C": [
                "Introduction to Programming",
                "Basic Structure of C Program",
                "Variables and Data Types",
                "Operators and Expressions",
                "Control Structures",
                "Functions and Recursion",
                "Arrays and Strings",
                "Pointers and Memory Management",
                "Structures and Unions",
                "File Operations"
            ],
            "Engineering Mathematics I": [
                "Complex Numbers",
                "Matrices and Determinants",
                "System of Linear Equations",
                "Sequences and Series",
                "Limits and Continuity",
                "Differentiation",
                "Applications of Derivatives",
                "Integration Techniques"
            ],
            "Engineering Physics": [
                "Mechanics and Motion",
                "Work, Energy and Power",
                "Oscillations and Waves",
                "Thermodynamics",
                "Optics and Light",
                "Modern Physics",
                "Atomic Structure",
                "Quantum Mechanics Basics"
            ],
            "Digital Logic": [
                "Number Systems",
                "Boolean Algebra",
                "Logic Gates",
                "Combinational Circuits",
                "Karnaugh Maps",
                "Sequential Circuits",
                "Flip-Flops and Latches",
                "Counters and Registers",
                "Memory Systems"
            ],
            "Basic Electrical Engineering": [
                "Circuit Fundamentals",
                "Ohm's Law and Kirchhoff's Laws",
                "Network Theorems",
                "AC Circuit Analysis",
                "Three-Phase Systems",
                "Magnetic Circuits",
                "Transformers",
                "Electrical Machines",
                "Measurement and Instrumentation"
            ],
            "Engineering Drawing I": [
                "Drawing Instruments and Materials",
                "Lettering and Dimensioning",
                "Geometric Constructions",
                "Orthographic Projections",
                "Isometric Drawings",
                "Sectional Views",
                "Auxiliary Views",
                "Development of Surfaces"
            ]
        }
        
    def _create_quiz_template(self) -> PromptTemplate:
        """Create a prompt template for quiz generation"""
        template = """
        Hi, your name is RAG Quiz Generator. You are a helpful and knowledgeable assistant trained to create quiz questions from the syllabus of the 1st semester at Deerwalk College.

        📚 Context Information:
        {context}

        🎯 Subject: {subject}
        📖 Chapter: {chapter}
        📊 Difficulty Level: {difficulty}
        🔢 Question Type: Multiple Choice Question

        🧠 Instructions:
        - Create ONE high-quality multiple choice question based on the context provided
        - Focus specifically on the chapter "{chapter}" from the subject "{subject}"
        - Make the question appropriate for {difficulty} level students
        - Provide 4 options (A, B, C, D) with only ONE correct answer
        - Include a detailed explanation for why the correct answer is right
        - Keep the language simple and student-friendly
        - Use examples or analogies if they help understanding
        - The question should be specific to the chapter content, not general subject knowledge

        📝 Output Format (JSON):
        {{
            "question": "Your clear and specific question about {chapter} here",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": 0,
            "explanation": "Detailed explanation of why this answer is correct, including relevant concepts from {chapter}",
            "difficulty": "{difficulty}",
            "topic": "{subject}",
            "chapter": "{chapter}"
        }}

        🚀 Generate the quiz question now:
        """
        
        return PromptTemplate(
            input_variables=["context", "subject", "chapter", "difficulty"],
            template=template
        )
    
    def _create_context_template(self) -> PromptTemplate:
        """Create a template for context-based responses"""
        template = """
        Hi, your name is RAG. You are a helpful and knowledgeable assistant trained to answer questions from the syllabus of the 1st semester of bachelor at electronics,communication and Information Engineering ,Institute of Engineering College.

        Context:
        {context}

        Instructions:
        - Focus only on topics from the official 1st semester syllabus OF bachelor at Electronics,Communication and Information Engineering,Institute of Engineering College
        - Cover concepts clearly and concisely, as a teacher would explain to a beginner student
        - Assume the user may be new to programming or computer science
        - Answer each question directly and clearly
        - Use examples or analogies if they help understanding
        - Keep responses focused, helpful, and easy to understand
        - Provide detailed explanations for complex concepts
        - If the question is about a specific chapter, focus on that chapter's content

        Question: {question}
        """
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    def _create_detailed_syllabus(self) -> List[Document]:
        """Create detailed chapter-wise syllabus content for BEI 1st Semester"""
        detailed_content = {
            "Programming in C": {
                "Introduction to Programming": """
                Introduction to Programming:
                - What is programming and why is it important?
                - History of programming languages
                - Overview of C programming language
                - Characteristics of C: structured, portable, efficient
                - Applications of C programming
                - Programming paradigms: procedural vs object-oriented
                - Problem-solving approach in programming
                """,
                "Basic Structure of C Program": """
                Basic Structure of C Program:
                - Preprocessor directives (#include, #define)
                - Header files and their purpose
                - Main function and its significance
                - Declaration section for variables
                - Executable statements
                - Comments in C (single-line and multi-line)
                - Compilation process: preprocessing, compilation, linking
                - Hello World program example
                """,
                "Variables and Data Types": """
                Variables and Data Types:
                - What are variables? Declaration and initialization
                - Data types: int, float, double, char
                - Size of data types using sizeof operator
                - Constant variables and literals
                - Type conversion: implicit and explicit
                - Scope of variables: local and global
                - Storage classes: auto, static, extern, register
                """,
                "Operators and Expressions": """
                Operators and Expressions:
                - Arithmetic operators (+, -, *, /, %)
                - Relational operators (<, >, <=, >=, ==, !=)
                - Logical operators (&&, ||, !)
                - Assignment operators (=, +=, -=, *=, /=)
                - Increment/decrement operators (++, --)
                - Bitwise operators (&, |, ^, ~, <<, >>)
                - Conditional operator (? :)
                - Operator precedence and associativity
                """,
                "Control Structures": """
                Control Structures:
                - Decision making: if, if-else, nested if-else
                - Switch-case statement with break and default
                - Loops: for, while, do-while
                - Loop control statements: break, continue, goto
                - Nested loops and their applications
                - Examples: factorial, fibonacci, prime numbers
                """,
                "Functions and Recursion": """
                Functions and Recursion:
                - Function definition and declaration
                - Function parameters and return values
                - Call by value and call by reference
                - Scope of variables in functions
                - Static variables in functions
                - Recursion: concept and implementation
                - Examples: factorial, fibonacci using recursion
                - Advantages and disadvantages of recursion
                """,
                "Arrays and Strings": """
                Arrays and Strings:
                - Array declaration and initialization
                - One-dimensional and multi-dimensional arrays
                - Array indexing and bounds checking
                - String representation as character arrays
                - String library functions: strlen, strcpy, strcat, strcmp
                - String input/output: gets, puts, scanf, printf
                - Array of strings
                """,
                "Pointers and Memory Management": """
                Pointers and Memory Management:
                - Pointer concept and declaration
                - Address operator (&) and dereference operator (*)
                - Pointer arithmetic
                - Arrays and pointers relationship
                - Dynamic memory allocation: malloc, calloc, realloc, free
                - Pointer to pointer (double pointer)
                - Function pointers
                """,
                "Structures and Unions": """
                Structures and Unions:
                - Structure definition and declaration
                - Accessing structure members
                - Array of structures
                - Nested structures
                - Pointer to structures
                - Union concept and usage
                - Difference between structures and unions
                - Enumeration (enum) data type
                """,
                "File Operations": """
                File Operations:
                - File handling concept
                - File opening modes: r, w, a, r+, w+, a+
                - File operations: fopen, fclose, fread, fwrite
                - Character I/O: fgetc, fputc
                - String I/O: fgets, fputs
                - Formatted I/O: fprintf, fscanf
                - File positioning: fseek, ftell, rewind
                """
            },
            "Digital Logic": {
                "Number Systems": """
                Number Systems:
                - Binary number system (base 2)
                - Decimal number system (base 10)
                - Octal number system (base 8)
                - Hexadecimal number system (base 16)
                - Conversion between number systems
                - Binary arithmetic: addition, subtraction, multiplication
                - Signed number representation: sign-magnitude, 1's complement, 2's complement
                - Binary codes: BCD, Gray code, ASCII
                """,
                "Boolean Algebra": """
                Boolean Algebra:
                - Boolean variables and constants
                - Basic Boolean operations: AND, OR, NOT
                - Boolean algebra laws: commutative, associative, distributive
                - De Morgan's theorems
                - Boolean function representation
                - Canonical forms: minterms and maxterms
                - Sum of Products (SOP) and Product of Sums (POS)
                - Boolean function simplification
                """,
                "Logic Gates": """
                Logic Gates:
                - AND gate: symbol, truth table, operation
                - OR gate: symbol, truth table, operation
                - NOT gate: symbol, truth table, operation
                - NAND gate: universal gate property
                - NOR gate: universal gate property
                - XOR gate: exclusive OR operation
                - XNOR gate: exclusive NOR operation
                - Gate-level circuit design
                """,
                "Combinational Circuits": """
                Combinational Circuits:
                - Half adder: design and truth table
                - Full adder: design and truth table
                - Binary adder circuits
                - Subtractor circuits
                - Multiplexer (MUX): 2:1, 4:1, 8:1
                - Demultiplexer (DEMUX)
                - Encoder and decoder circuits
                - Code converters
                """,
                "Karnaugh Maps": """
                Karnaugh Maps:
                - K-map concept and construction
                - 2-variable K-map
                - 3-variable K-map
                - 4-variable K-map
                - Grouping rules in K-map
                - Don't care conditions
                - Prime implicants and essential prime implicants
                - Minimization using K-map
                """,
                "Sequential Circuits": """
                Sequential Circuits:
                - Difference between combinational and sequential circuits
                - Clock signals and synchronization
                - State concept in sequential circuits
                - State tables and state diagrams
                - Mealy and Moore machines
                - Analysis of sequential circuits
                - Design of sequential circuits
                """,
                "Flip-Flops and Latches": """
                Flip-Flops and Latches:
                - SR latch: operation and truth table
                - Gated SR latch
                - D latch: operation and applications
                - SR flip-flop: clocked operation
                - D flip-flop: data storage element
                - JK flip-flop: no invalid state
                - T flip-flop: toggle operation
                - Master-slave flip-flops
                """,
                "Counters and Registers": """
                Counters and Registers:
                - Asynchronous counters (ripple counters)
                - Synchronous counters
                - Up counters and down counters
                - Modulo-N counters
                - Shift registers: SISO, SIPO, PISO, PIPO
                - Ring counters
                - Johnson counters
                - Applications of counters and registers
                """,
                "Memory Systems": """
                Memory Systems:
                - Memory organization and addressing
                - ROM (Read-Only Memory): PROM, EPROM, EEPROM
                - RAM (Random Access Memory): SRAM, DRAM
                - Memory expansion techniques
                - Memory interfacing
                - Programmable logic devices: PLA, PAL, FPGA
                - Memory hierarchy concept
                """
            },
            "Basic Electrical Engineering": {
                "Circuit Fundamentals": """
                Circuit Fundamentals:
                - Electric charge, current, and voltage
                - Power and energy in electrical circuits
                - Passive circuit elements: resistor, inductor, capacitor
                - Active circuit elements: voltage source, current source
                - Sign conventions for voltage and current
                - Circuit diagrams and symbols
                - Basic circuit terminology: node, branch, loop, mesh
                """,
                "Ohm's Law and Kirchhoff's Laws": """
                Ohm's Law and Kirchhoff's Laws:
                - Ohm's law: V = I × R
                - Resistance and conductance
                - Kirchhoff's Current Law (KCL)
                - Kirchhoff's Voltage Law (KVL)
                - Series and parallel resistor combinations
                - Voltage divider and current divider rules
                - Applications and problem solving
                """,
                "Network Theorems": """
                Network Theorems:
                - Thevenin's theorem: concept and applications
                - Norton's theorem: concept and applications
                - Thevenin-Norton equivalent transformations
                - Superposition theorem
                - Maximum power transfer theorem
                - Millman's theorem
                - Reciprocity theorem
                - Substitution theorem
                """,
                "AC Circuit Analysis": """
                AC Circuit Analysis:
                - AC waveforms: sine, cosine, square, triangular
                - RMS and average values
                - Phasor representation
                - Impedance and admittance
                - AC analysis of R, L, C circuits
                - Power in AC circuits: real, reactive, apparent
                - Power factor and power factor correction
                - Resonance in AC circuits
                """,
                "Three-Phase Systems": """
                Three-Phase Systems:
                - Generation of three-phase voltages
                - Balanced three-phase systems
                - Star (Y) connection
                - Delta (Δ) connection
                - Line and phase voltages/currents
                - Power in three-phase systems
                - Advantages of three-phase systems
                - Three-phase power measurement
                """,
                "Magnetic Circuits": """
                Magnetic Circuits:
                - Magnetic field and flux
                - Magnetic materials: ferromagnetic, paramagnetic, diamagnetic
                - B-H curve and hysteresis
                - Magnetic circuit analysis
                - Ampere's law
                - Reluctance and permeance
                - Magnetic circuit with air gap
                - Electromagnets and their applications
                """,
                "Transformers": """
                Transformers:
                - Transformer principle of operation
                - Ideal transformer theory
                - Transformer construction
                - EMF equation of transformer
                - Transformer on no-load and on-load
                - Equivalent circuit of transformer
                - Transformer losses and efficiency
                - Types of transformers and applications
                """,
                "Electrical Machines": """
                Electrical Machines:
                - DC generators: principle and types
                - EMF equation of DC generator
                - DC motors: principle and types
                - Torque equation of DC motor
                - Three-phase induction motors
                - Rotating magnetic field concept
                - Slip and rotor frequency
                - Torque-speed characteristics
                """,
                "Measurement and Instrumentation": """
                Measurement and Instrumentation:
                - Measurement systems and standards
                - Errors in measurements
                - Moving coil instruments
                - Moving iron instruments
                - Dynamometer instruments
                - Measurement of voltage, current, power
                - Wattmeter connections
                - Energy measurement
                """
            }
        }
        
        documents = []
        for subject, chapters in detailed_content.items():
            for chapter, content in chapters.items():
                doc = Document(
                    page_content=content,
                    metadata={
                        "subject": subject,
                        "chapter": chapter,
                        "difficulty": "Intermediate"
                    }
                )
                documents.append(doc)
        
        return documents
    
    def initialize_vectorstore(self):
        """Initialize the vector store with detailed chapter-wise data"""
        try:
            documents = self._create_detailed_syllabus()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            self.vectorstore = FAISS.from_documents(
                split_docs,
                self.embeddings
            )
            logger.info(f"✅ Created vector store with {len(split_docs)} chunks")
            return True
        except Exception as e:
            logger.error(f"❌ Error creating vector store: {e}")
            return False
    
    def retrieve_relevant_context(self, query: str, subject: str, chapter: str, k: int = 3) -> str:
        """Retrieve relevant context for a specific chapter"""
        if not self.vectorstore:
            raise ValueError("Vector store not created.")
        
        try:
            # Enhanced query with subject and chapter context
            enhanced_query = f"{subject} {chapter} {query}"
            docs = self.vectorstore.similarity_search(enhanced_query, k=k)
            
            # Filter documents by subject and chapter if possible
            relevant_docs = []
            for doc in docs:
                if (doc.metadata.get('subject') == subject and 
                    doc.metadata.get('chapter') == chapter):
                    relevant_docs.append(doc)
            
            # If no exact matches, use all retrieved docs
            if not relevant_docs:
                relevant_docs = docs
            
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            return context
        except Exception as e:
            logger.error(f"❌ Error retrieving context: {e}")
            return "Context not available"
    
    def generate_quiz_question(self, subject: str, chapter: str, difficulty: str = "Beginner") -> QuizQuestion:
        """Generate a single quiz question for a specific chapter"""
        query = f"{chapter} concepts fundamentals"
        
        try:
            # Retrieve relevant context for the specific chapter
            context = self.retrieve_relevant_context(query, subject, chapter)
            
            # Create the chain
            rag_chain = self.quiz_template | self.llm
            
            # Generate the question
            response = rag_chain.invoke({
                "context": context,
                "subject": subject,
                "chapter": chapter,
                "difficulty": difficulty
            })
            
            response_text = str(response.content) if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                question_data = json.loads(json_str)
                
                return QuizQuestion(
                    question=question_data["question"],
                    options=question_data["options"],
                    correct_answer=question_data["correct_answer"],
                    explanation=question_data["explanation"],
                    difficulty=question_data["difficulty"],
                    topic=question_data["topic"],
                    chapter=question_data["chapter"]
                )
            else:
                return self._create_fallback_question(subject, chapter, difficulty)
                
        except Exception as e:
            logger.error(f"❌ Error generating question: {e}")
            return self._create_fallback_question(subject, chapter, difficulty)
    
    def _create_fallback_question(self, subject: str, chapter: str, difficulty: str) -> QuizQuestion:
        """Create a fallback question if generation fails"""
        fallback_questions = {
            "Programming in C": {
                "Introduction to Programming": {
                    "question": "What is the primary characteristic of C programming language?",
                    "options": ["Object-oriented", "Structured and procedural", "Functional", "Logic-based"],
                    "correct_answer": 1,
                    "explanation": "C is primarily a structured and procedural programming language, emphasizing functions and structured programming concepts."
                },
                "Variables and Data Types": {
                    "question": "Which data type is used to store a single character in C?",
                    "options": ["int", "float", "char", "string"],
                    "correct_answer": 2,
                    "explanation": "The 'char' data type is used to store a single character in C programming."
                }
            },
            "Digital Logic": {
                "Number Systems": {
                    "question": "What is the binary equivalent of decimal number 8?",
                    "options": ["1000", "1010", "1100", "1110"],
                    "correct_answer": 0,
                    "explanation": "Decimal 8 in binary is 1000 (1×2³ + 0×2² + 0×2¹ + 0×2⁰ = 8)"
                },
                "Logic Gates": {
                    "question": "Which gate is known as the universal gate?",
                    "options": ["AND", "OR", "NOT", "NAND"],
                    "correct_answer": 3,
                    "explanation": "NAND gate is called universal gate because any Boolean function can be implemented using only NAND gates."
                }
            }
        }
        
        # Try to find specific chapter fallback
        if subject in fallback_questions and chapter in fallback_questions[subject]:
            fb = fallback_questions[subject][chapter]
            return QuizQuestion(
                question=fb["question"],
                options=fb["options"],
                correct_answer=fb["correct_answer"],
                explanation=fb["explanation"],
                difficulty=difficulty,
                topic=subject,
                chapter=chapter
            )
        
        # Default fallback
        return QuizQuestion(
            question=f"What is a key concept in {chapter} of {subject}?",
            options=["Option A", "Option B", "Option C", "Option D"],
            correct_answer=0,
            explanation=f"This is a fallback question for {chapter} in {subject}.",
            difficulty=difficulty,
            topic=subject,
            chapter=chapter
        )
    
    def generate_quiz(self, subject: str, chapter: str, num_questions: int = 5, difficulty: str = "Beginner") -> List[QuizQuestion]:
        """Generate a complete quiz for a specific chapter"""
        questions = []
        
        for i in range(num_questions):
            try:
                question = self.generate_quiz_question(subject=subject, chapter=chapter, difficulty=difficulty)
                questions.append(question)
                logger.info(f"Generated question {i+1}/{num_questions} for {chapter}")
            except Exception as e:
                logger.error(f"Error generating question {i+1}: {e}")
                questions.append(self._create_fallback_question(subject, chapter, difficulty))
        
        return questions
    
    def get_chapters(self, subject: str) -> List[str]:
        """Get available chapters for a subject"""
        return self.chapter_structure.get(subject, [])
    
    def get_subjects(self) -> List[str]:
        """Get available subjects"""
        return list(self.chapter_structure.keys())
    
    def ask_question(self, question: str, subject: str = None, chapter: str = None) -> str:
        """Answer a question using RAG with optional subject/chapter context"""
        try:
            if subject and chapter:
                context = self.retrieve_relevant_context(question, subject, chapter)
            else:
                # General context retrieval
                docs = self.vectorstore.similarity_search(question, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
            
            rag_chain = self.context_template | self.llm
            
            response = rag_chain.invoke({
                "context": context,
                "question": question
            })
            
            return str(response.content) if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f" Error: {str(e)}"

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
CORS(app)

# Global variables
quiz_generator = None
active_sessions = {}

@app.before_request
def initialize_app():
    """Initialize the RAG system before handling requests"""
    global quiz_generator
    if quiz_generator is None:
        logger.info("Initializing RAG Quiz Generator...")
        quiz_generator = DeerWalkRAGQuizGenerator()
        success = quiz_generator.initialize_vectorstore()
        if success:
            logger.info("RAG system initialized successfully")
        else:
            logger.error(" Failed to initialize RAG system")


@app.route('/api/subjects', methods=['GET'])
def get_subjects():
    """Get available subjects"""
    subjects = quiz_generator.get_subjects()
    return jsonify({'subjects': subjects})

@app.route('/api/chapters', methods=['GET'])
def get_chapters():
    """Get available chapters for a subject"""
    subject = request.args.get('subject')
    if not subject:
        return jsonify({'success': False, 'error': 'Subject parameter required'})
    
    chapters = quiz_generator.get_chapters(subject)
    return jsonify({'success': True, 'chapters': chapters})


@app.route('/api/generate-quiz', methods=['POST'])
def generate_quiz():
    """Generate a new chapter-wise quiz"""
    try:
        data = request.json
        subject = data.get('subject')
        chapter = data.get('chapter')
        difficulty = data.get('difficulty', 'Beginner')
        num_questions = data.get('num_questions', 5)
        
        if not subject or not chapter:
            return jsonify({'success': False, 'error': 'Subject and chapter are required'})
        
        # Validate subject and chapter
        if subject not in quiz_generator.get_subjects():
            return jsonify({'success': False, 'error': 'Invalid subject'})
        
        if chapter not in quiz_generator.get_chapters(subject):
            return jsonify({'success': False, 'error': 'Invalid chapter for the selected subject'})
        
        # Generate quiz questions
        questions = quiz_generator.generate_quiz(subject, chapter, num_questions, difficulty)
        
        # Create a new session
        session_id = str(uuid.uuid4())
        quiz_session = QuizSession(
            session_id=session_id,
            questions=questions,
            subject=subject,
            chapter=chapter
        )
        
        # Store session
        active_sessions[session_id] = quiz_session
        
        # Return session info and first question
        return jsonify({
            'success': True,
            'session_id': session_id,
            'subject': subject,
            'chapter': chapter,
            'total_questions': len(questions),
            'current_question': 0,
            'question_data': {
                'question': questions[0].question,
                'options': questions[0].options,
                'topic': questions[0].topic,
                'chapter': questions[0].chapter,
                'difficulty': questions[0].difficulty
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/submit-answer', methods=['POST'])
def submit_answer():
    """Submit an answer for the current question"""
    try:
        data = request.json
        session_id = data.get('session_id')
        user_answer = data.get('answer')
        
        if session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        quiz_session = active_sessions[session_id]
        current_question = quiz_session.questions[quiz_session.current_question]
        
        # Check if answer is correct
        is_correct = user_answer == current_question.correct_answer
        if is_correct:
            quiz_session.score += 1
        
        # Store user's answer
        quiz_session.answers.append(user_answer)
        
        # Prepare response
        response_data = {
            'success': True,
            'is_correct': is_correct,
            'correct_answer': current_question.correct_answer,
            'explanation': current_question.explanation,
            'current_score': quiz_session.score,
            'question_number': quiz_session.current_question + 1,
            'total_questions': len(quiz_session.questions)
        }
        
        # Move to next question
        quiz_session.current_question += 1
        
        # Check if quiz is completed
        if quiz_session.current_question >= len(quiz_session.questions):
            quiz_session.completed = True
            response_data['quiz_completed'] = True
            response_data['final_score'] = quiz_session.score
            response_data['percentage'] = (quiz_session.score / len(quiz_session.questions)) * 100
        else:
            # Return next question
            next_question = quiz_session.questions[quiz_session.current_question]
            response_data['next_question'] = {
                'question': next_question.question,
                'options': next_question.options,
                'topic': next_question.topic,
                'difficulty': next_question.difficulty
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error submitting answer: {e}")
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/get-quiz-history', methods=['GET'])
def get_quiz_history():
    """Get quiz history for the current session"""
    try:
        session_id = request.args.get('session_id')
        
        if session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Session not found'})
        
        quiz_session = active_sessions[session_id]
        
        # Prepare detailed results
        results = []
        for i, question in enumerate(quiz_session.questions[:len(quiz_session.answers)]):
            user_answer = quiz_session.answers[i]
            results.append({
                'question_number': i + 1,
                'question': question.question,
                'options': question.options,
                'user_answer': user_answer,
                'correct_answer': question.correct_answer,
                'is_correct': user_answer == question.correct_answer,
                'explanation': question.explanation,
                'topic': question.topic,
                'difficulty': question.difficulty
            })
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'total_questions': len(quiz_session.questions),
            'completed_questions': len(quiz_session.answers),
            'score': quiz_session.score,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error getting quiz history: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Get available topics for BEI 1st Semester"""
    topics = [
        'Programming in C',
        'Engineering Mathematics I',
        'Engineering Physics',
        'Digital Logic',
        'Basic Electrical Engineering',
        'Engineering Drawing I'
    ]
    return jsonify({'topics': topics})

@app.route('/api/difficulties', methods=['GET'])
def get_difficulties():
    """Get available difficulty levels"""
    difficulties = ['Beginner', 'Intermediate', 'Advanced']
    return jsonify({'difficulties': difficulties})

# Signup route with password hashing
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    try:
        if request.is_json:
            data = request.get_json()
            email = data.get("email")
            password = data.get("password")
        else:
            email = request.form.get("email")
            password = request.form.get("password")

        if not email or not password:
            return jsonify({"success": False, "error": "Email and password are required"}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cur.fetchone()
        if existing_user:
            cur.close()
            conn.close()
            return jsonify({"success": False, "error": "User already exists"}), 409

        # Hash the password before storing it
        hashed_password = generate_password_hash(password)

        cur.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (email, hashed_password))
        conn.commit()
        cur.close()
        conn.close()

        if request.is_json:
            return jsonify({"success": True, "message": "User registered successfully"}), 201
        else:
            return render_template("index.html", message="User registered successfully. Please log in.")

    except Exception as e:
        logger.error(f"Error in signup: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# Login route verifying hashed password


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/engineering")
def engineering():
    return render_template("engineering.html") 

@app.route("/semester")
def semester():
    return render_template("semester.html") 
@app.route("/subject")
def subject():
    return render_template("subject.html") 
@app.route("/physics")
def quiz():
    return render_template("physics.html") 

   
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Missing JSON data"}), 400
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"success": False, "error": "Email and password required"}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()

    if user is None:
        return jsonify({"success": False, "error": "User not found"}), 404

    # Plain-text password comparison
    if password == user["password_hash"]:
        user_data = dict(user)
        user_data.pop("password_hash", None)
        return jsonify({"success": True, "message": "Login successful", "user": user_data}), 200
    else:
        return jsonify({"success": False, "error": "Invalid password"}), 401
@app.route("/quiz")
def quiz_page():
    return render_template("quiz.html")

gemini = genai.GenerativeModel("gemini-1.5-flash")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        prompt = (
            "You are an expert AI assistant for all academic subjects including Physics, Chemistry, "
            "Mathematics, and Engineering. Answer the following question accurately and concisely.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        response = gemini.generate_content(prompt)
        answer = response.text.strip()

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)