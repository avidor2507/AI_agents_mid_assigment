"""
PDF loading and parsing module.

This module handles loading PDF documents and extracting structured information
including text content, document structure (sections, headers), and metadata
(timestamps, dates, entities).

Follows Single Responsibility Principle - this module only handles PDF loading.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from pypdf import PdfReader
from src.utils.exceptions import PDFLoadingError
from src.utils.logger import logger
from src.utils.helpers import parse_timestamp, extract_entities
import re


class Document:
    """
    Represents a loaded PDF document with structure and metadata.
    
    This class encapsulates a document with:
    - Raw text content
    - Document structure (sections, headers)
    - Extracted metadata (timestamps, entities, dates)
    """
    
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        """
        Initialize a Document instance.
        
        Args:
            text: Full text content of the document
            metadata: Dictionary containing document metadata
        """
        self.text = text
        self.metadata = metadata or {}
        self.sections: List[Dict[str, Any]] = []
        self.pages: List[str] = []
    
    def add_section(self, section: Dict[str, Any]):
        """Add a section to the document structure."""
        self.sections.append(section)
    
    def add_page(self, page_text: str):
        """Add a page to the document."""
        self.pages.append(page_text)


class PDFLoader:
    """
    Loader for PDF documents with text extraction and structure parsing.
    
    Responsibilities:
    - Extract text from PDF files
    - Parse document structure (sections, headers)
    - Extract metadata (timestamps, dates, entities)
    - Return structured Document object
    
    Follows Single Responsibility Principle - only handles PDF loading.
    """
    
    def __init__(self):
        """Initialize the PDF loader."""
        self.logger = logger
    
    def load(self, file_path: Path) -> Document:
        """
        Load and parse a PDF file.
        
        This method:
        1. Reads the PDF file
        2. Extracts text from all pages
        3. Identifies sections and headers
        4. Extracts metadata (timestamps, entities)
        5. Returns a structured Document object
        
        Args:
            file_path: Path to the PDF file
        
        Returns:
            Document: Structured document object with text and metadata
        
        Raises:
            PDFLoadingError: If PDF cannot be loaded or parsed
        """
        try:
            self.logger.info(f"Loading PDF from: {file_path}")
            
            # Verify file exists
            if not file_path.exists():
                raise PDFLoadingError(f"PDF file not found: {file_path}")
            
            # Read PDF
            reader = PdfReader(str(file_path))
            
            # Extract text from all pages
            pages_text = []
            full_text = ""
            
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                pages_text.append(page_text)
                full_text += page_text + "\n"
            
            self.logger.info(f"Extracted text from {len(pages_text)} pages")
            
            # Create document object
            document = Document(text=full_text)
            document.pages = pages_text
            
            # Extract metadata
            metadata = self._extract_metadata(full_text, file_path)
            document.metadata = metadata
            
            # Parse document structure (sections, headers)
            sections = self._parse_structure(full_text)
            for section in sections:
                document.add_section(section)
            
            self.logger.info(f"Successfully loaded PDF with {len(sections)} sections")
            
            return document
        
        except Exception as e:
            error_msg = f"Error loading PDF {file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise PDFLoadingError(error_msg) from e
    
    def _extract_metadata(self, text: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from document text.
        
        Extracts:
        - Timestamps and dates
        - Entities (money, emails, phones)
        - File information
        
        Args:
            text: Full document text
            file_path: Path to the PDF file
        
        Returns:
            Dict[str, Any]: Dictionary of extracted metadata
        """
        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
        }
        
        # Extract timestamps
        timestamps = []
        # Look for timestamp patterns in the text
        lines = text.split('\n')
        for line in lines:
            timestamp = parse_timestamp(line)
            if timestamp:
                timestamps.append(timestamp.isoformat())
        
        if timestamps:
            metadata["timestamps"] = timestamps
            metadata["first_timestamp"] = timestamps[0]
            metadata["last_timestamp"] = timestamps[-1]
        
        # Extract entities
        entities = extract_entities(text)
        if any(entities.values()):  # If any entities were found
            metadata["entities"] = entities
        
        # Try to extract claim number or ID (common patterns)
        claim_patterns = [
            r'[Cc]laim\s*[#:]?\s*(\d+)',
            r'[Cc]ase\s*[#:]?\s*(\d+)',
            r'[Ii][Dd]:\s*(\d+)',
        ]
        
        for pattern in claim_patterns:
            import re
            match = re.search(pattern, text)
            if match:
                metadata["claim_id"] = match.group(1)
                break
        
        return metadata
    
    @staticmethod
    def _parse_structure(text: str) -> List[Dict[str, Any]]:
        """
        Parse document structure (sections, headers).
        
        This is a simple implementation that identifies sections by:
        - Headers (text in all caps or numbered sections)
        - Line breaks followed by capitalized lines
        
        For more sophisticated parsing, consider using NLP models.
        
        Args:
            text: Full document text
        
        Returns:
            List[Dict[str, Any]]: List of section dictionaries with structure
        """
        sections: List[Dict[str, Any]] = []
        lines = text.split('\n')

        current_section: Optional[Dict[str, Any]] = None
        section_text: List[str] = []
        section_num = 0

        for i, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue

            # ------------------------------------------------------------------
            # 1) Strong rule: explicit "Section X – ..." style headings
            # ------------------------------------------------------------------
            # Example: "Section 3 – Detailed Chronological Timeline of Events"
            section_match = re.match(r'^Section\s+(\d+)\s*[–-]\s*(.*)$', line)

            if section_match:
                # Save previous section if exists
                if current_section and section_text:
                    current_section["text"] = "\n".join(section_text)
                    current_section["line_count"] = len(section_text)
                    sections.append(current_section)

                section_num_in_text = int(section_match.group(1))
                header_text = line  # Keep full header line as-is

                current_section = {
                    "section_id": f"section_{section_num_in_text}",
                    "header": header_text,
                    "section_number": section_num_in_text,
                    "start_line": i,
                }
                section_text = []
                # We continue to next line (content of this section)
                continue

            # ------------------------------------------------------------------
            # 2) Fallback heuristic for other headers (e.g., main title)
            # ------------------------------------------------------------------
            is_header = (
                len(line) < 80 and
                (line.isupper() or line.istitle()) and
                (i == 0 or not lines[i - 1].strip())  # Previous line was empty
            )

            numbered_section = bool(
                re.match(r'^(\d+[\.\)]\s*|Section\s+\d+:)', line, re.IGNORECASE)
            )

            if is_header or numbered_section:
                # Save previous section if exists
                if current_section and section_text:
                    current_section["text"] = "\n".join(section_text)
                    current_section["line_count"] = len(section_text)
                    sections.append(current_section)

                # Special case for very first header at top of document:
                # treat it as section_0 so it does not collide with "Section 1 – ..."
                if section_num == 0 and i == 0:
                    section_num = 0
                    section_id = "section_0"
                    section_number = 0
                else:
                    section_num += 1
                    section_id = f"section_{section_num}"
                    section_number = section_num

                current_section = {
                    "section_id": section_id,
                    "header": line,
                    "section_number": section_number,
                    "start_line": i,
                }
                section_text = []
            else:
                # Normal content line -> add to current section
                section_text.append(line)

        # Add last section
        if current_section and section_text:
            current_section["text"] = "\n".join(section_text)
            current_section["line_count"] = len(section_text)
            sections.append(current_section)

        # If no sections found, create one section with all text
        if not sections:
            sections.append(
                {
                    "section_id": "section_1",
                    "header": "Main Content",
                    "section_number": 1,
                    "text": text,
                    "line_count": len(lines),
                    "start_line": 0,
                }
            )

        return sections



