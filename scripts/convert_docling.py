import argparse
import logging
import os
import sys

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
except Exception as e:
    print(f"CRITICAL: Failed to import docling. Error: {e}")
    print("Please ensure it is installed correctly: pip install docling")
    sys.exit(1)

# Configure hospital-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("docling_converter")

def convert_document(input_path: str, output_path: str, output_format: str):
    """
    Converts a document (PDF, HTML, DOCX, etc.) into an LLM-optimized format (Markdown).
    
    Args:
        input_path (str): The absolute or relative path to the input document.
        output_path (str): The destination path for the converted output.
        output_format (str): The requested output format. Only 'md' is fully supported here.
    """
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
        
    logger.info(f"Targeting input document: {input_path}")
    logger.info(f"Requested output format: {output_format.upper()}")

    # Determine fallback if user passed an ipynb directly
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.ipynb':
        logger.warning(
            "Direct ingestion of raw .ipynb files via Docling may yield sub-optimal results or fail entirely "
            "as Docling is heavily optimized for rendering layout-driven formats (PDF, DOCX, HTML). "
            "For rigorous context extraction, it is highly recommended to PDF the notebook first."
        )

    logger.info("Initializing Docling DocumentConverter with Image Extraction...")
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    except Exception as e:
        logger.error(f"Failed to initialize DocumentConverter: {str(e)}")
        sys.exit(1)

    logger.info("Commencing conversion process. This may take a moment depending on document complexity...")
    try:
        conversion_result = converter.convert(input_path)
    except Exception as e:
        logger.error(f"Docling encountered a fatal error during conversion: {str(e)}")
        sys.exit(1)
        
    logger.info("Conversion successful. Extracting output payload...")

    try:
        if output_format == 'md':
            output_data = conversion_result.document.export_to_markdown()
        else:
            logger.error(f"Unsupported output format requested: {output_format}")
            sys.exit(1)
            
        # 1. Save the Markdown
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_data)
        logger.info(f"Successfully wrote payload to: {output_path}")

        # 2. Extract and save the images
        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        image_dir = os.path.join(output_dir, f"{base_name}_images")

        img_count = 0
        for element, _level in conversion_result.document.iterate_items():
            if element.label == "picture" and hasattr(element, "image") and element.image:
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                    
                img_path = os.path.join(image_dir, f"figure_{img_count}.png")
                element.image.pil_image.save(img_path)
                logger.info(f"Extracted image {img_count} -> {img_path}")
                img_count += 1
                
        if img_count > 0:
            logger.info(f"Successfully extracted {img_count} images into {image_dir}")
        else:
            logger.info("No extractable images/figures were found in the document.")
        
    except IOError as e:
        logger.error(f"I/O Error while writing to {output_path}: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error formatting/writing output: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Robust document conversion script utilizing IBM's Docling for generating structured LLM context.",
        epilog="Designed for high-fidelity extraction of academic/research documents (especially PDFs)."
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to the input document (e.g., notebooks/research_report_v3.pdf)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Destinaton path for the converted file (e.g., notebooks/research_report_v3.md)'
    )
    
    parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['md'],
        default='md',
        help="Target output format. Only Markdown ('md') is currently optimized for LLM ingestion."
    )

    args = parser.parse_args()
    
    # Ensure input and output paths are absolute or correctly formatted relative
    input_abspath = os.path.abspath(args.input)
    output_abspath = os.path.abspath(args.output)
    
    convert_document(input_abspath, output_abspath, args.format)

if __name__ == '__main__':
    main()
