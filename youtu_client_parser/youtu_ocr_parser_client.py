import os
import json
import base64
import argparse
from openai import OpenAI
from typing import List, Dict, Union, Optional
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from youtu_parsing_utils.consts import IMAGE_EXT, PDF_EXT
from youtu_parsing_utils.prompt import PROMPT_DICT
from youtu_parsing_utils.layout_utils import parse_layout_str
from youtu_parsing_utils.table_utils import convert_table_ostl_to_html
from youtu_parsing_utils.image_utils import load_images_from_pdf, load_image
from youtu_parsing_utils.figtext_utils import parse_figtext_str
from youtu_parsing_utils.visualize_utils import draw_layout_on_image
from youtu_parsing_utils.parser_utils import (
    box_to_str, clean_repeated_substrings,
    parse_batch_results, categorize_layout_items,
    build_hierarchy_json
)


class ServiceClient:
    """
    Service client for handling API communication via OpenAI-compatible API.
    
    This class provides a clean interface for communicating with vision-language models
    through OpenAI-compatible API endpoints.
    """
    
    def __init__(self, base_url: str = "http://localhost:8080/v1", model_name: str = "utuv1"):
        """
        Initialize the service client.
        
        Args:
            base_url: Server OpenAI-compatible API URL (should end with /v1)
            model_name: Model name to use for API calls
        """
        self.base_url = base_url
        self.model_name = model_name
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="not-needed"
        )
        print(f"✅ Initialized service client with model: {model_name}")

    def _encode_image(self, image: Union[str, Path, Image.Image]) -> str:
        """Encode image to base64 string."""
        # Handle file path input
        if isinstance(image, (str, Path)):
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        # Handle PIL Image object
        elif isinstance(image, Image.Image):
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            raise ValueError("Image must be file path or PIL Image object")
    
    def chat(self,
             prompt: str,
             image: Optional[Union[str, Path, Image.Image]] = None,
             system: Optional[str] = None,
             temperature: float = 0.1,
             top_p: float = 0.9) -> str:
        """
        Chat with vision-language model via OpenAI-compatible API.
        
        Args:
            prompt: User prompt text
            image: Image file path or PIL Image object (optional)
            system: System prompt (optional)
            temperature: Temperature parameter for generation
            top_p: Top-p parameter for generation
            
        Returns:
            Model response text
        """
        # Build messages
        messages = []
        
        # Add system message if provided
        if system:
            messages.append({
                "role": "system",
                "content": system
            })
        
        # Build user message content
        content = []
        
        # Add image if provided
        if image is not None:
            base64_image = self._encode_image(image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            })
        content.append({"type": "text", "text": prompt})
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        # Send request via OpenAI client
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error communicating with service: {e}")
            return ""


class YoutuOCRParserClient:
    
    def __init__(self, 
                 base_url: str = "http://localhost:8080/v1",
                 model_name: str = "utuv1"):
        """
        Initialize the OCR parser service with service client and configuration.
        
        Args:
            base_url: Server OpenAI-compatible API URL (should end with /v1)
            model_name: Model name to use for API calls
        """
        self.client = ServiceClient(base_url, model_name)

    def _get_prompt(self, prompt_mode: str, bbox: Optional[List[float]] = None) -> str:
        prompt = PROMPT_DICT[prompt_mode]
        
        region_specific_modes = ["text_recognize", "title_recognize", "table_recognize", 
                                 "chart_logic_recognize", "chart_data_recognize"]
        if prompt_mode in region_specific_modes:
            assert bbox is not None, f"Bounding box required for {prompt_mode}"
            # Format prompt with bbox coordinates
            prompt = prompt.format(bbox[0], bbox[1], bbox[2], bbox[3])
        
        return prompt

    def _inference(self, image: Image.Image, prompt: str) -> str:
        """
        Perform inference with the service client.
        
        Args:
            image: Input PIL Image
            prompt: Text prompt for the model
            
        Returns:
            Model response text
        """
        try:
            return self.client.chat(prompt=prompt, image=image)
        except Exception as e:
            print(f"Service inference error: {e}")
            return ""

    def _infer_bbox_query_batch(self,
                               image: Image.Image, 
                               bbox_list: List[List[float]], 
                               layout_types: List[str], 
                               max_batch_size: int = 5) -> List[str]:
        if not bbox_list:
            return []
        
        # Limit batch size
        bbox_list = bbox_list[:max_batch_size]
        layout_types = layout_types[:max_batch_size]
        
        # Handle single bbox case
        if len(bbox_list) == 1:
            prompt_box = self._get_prompt("text_recognize", bbox_list[0])
            result = self._inference(image, prompt_box)
            return [result]
        
        # Handle multiple bboxes
        box_query_parts = [box_to_str(bbox, layout_type) 
                          for bbox, layout_type in zip(bbox_list, layout_types)]
        
        combined_query = (
            "Based on the given input field coordinates and layout type, "
            "identify and extract the content within the specified region. "
            "Formulas shall be represented in LaTeX notation, and tables "
            "shall be structured in OTSL format: " + "|".join(box_query_parts)
        )
        
        print(f"Batch query: {combined_query}")
        output_text = self._inference(image, combined_query)
        print(f"Batch query result: {output_text}")
        
        # Parse model output into individual results
        return parse_batch_results(output_text, len(bbox_list))

    def _process_figure_items(self,
                             image: Image.Image, 
                             layout_items: List[List], 
                             figure_indices: List[int], 
                             recognize_results: List[str]) -> None:
        for i in figure_indices:
            try:
                x1, y1, x2, y2, layout_type = layout_items[i]
                # Crop figure region from image
                crop_image = image.crop([int(x1), int(y1), int(x2), int(y2)])
                
                # Recognize text in figure
                prompt_box = self._get_prompt("figure_recognize")
                response_box_content = self._inference(crop_image, prompt_box)
                response_box_content = clean_repeated_substrings(
                    response_box_content) if response_box_content else ""

                # Parse structured figure text items
                figtext_items = parse_figtext_str(response_box_content)
                
                # Combine all figure text content
                recognize_results[i] = ''.join(item['content'] + '\n' for item in figtext_items)
            except Exception as e:
                print(f"FIGURE crop recognition error: {e}")
                recognize_results[i] = ""
    
    def _process_chart_items(self,
                            image: Image.Image, 
                            layout_items: List[List], 
                            chart_indices: List[int], 
                            recognize_results: List[str]) -> None:
        for i in chart_indices:
            try:
                x1, y1, x2, y2, layout_type = layout_items[i]
                # Generate chart-specific prompt with bbox
                prompt_box = self._get_prompt(f"{layout_type.lower()}_recognize", [x1, y1, x2, y2])
                recognize_results[i] = self._inference(image, prompt_box)
            except Exception as e:
                print(f"CHART recognition error: {e}")
                recognize_results[i] = ""

    def _process_table_items(self,
                            image: Image.Image, 
                            layout_items: List[List], 
                            table_indices: List[int], 
                            recognize_results: List[str]) -> None:
        """Process table items and convert to HTML format."""
        for i in table_indices:
            try:
                x1, y1, x2, y2, layout_type = layout_items[i]
                # Recognize table structure in OTSL format
                prompt_box = self._get_prompt("table_recognize", [x1, y1, x2, y2])
                response_box_content = self._inference(image, prompt_box)
                
                # Convert OTSL to HTML
                recognize_results[i] = convert_table_ostl_to_html(response_box_content) if response_box_content else ""
            except Exception as e:
                print(f"TABLE recognition or HTML conversion error: {e}")
                recognize_results[i] = ""

    def _process_ocr_items_batch(self,
                                image: Image.Image, 
                                layout_items: List[List], 
                                ocr_indices: List[int], 
                                recognize_results: List[str], 
                                batch_size: int = 5) -> None:
        for batch_start in range(0, len(ocr_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(ocr_indices))
            batch_indices = ocr_indices[batch_start:batch_end]
            
            # Extract bounding boxes and layout types for current batch
            batch_bboxes = [[*layout_items[idx][:4]] for idx in batch_indices]
            batch_types = [layout_items[idx][4] for idx in batch_indices]
            
            try:
                # Perform batch OCR inference on current batch
                batch_results = self._infer_bbox_query_batch(
                    image, batch_bboxes, batch_types, max_batch_size=batch_size
                )
                
                # Validate that result count matches input count to ensure data integrity
                if len(batch_results) != len(batch_bboxes):
                    print(f"Warning: Batch result count ({len(batch_results)}) "
                          f"doesn't match input count ({len(batch_bboxes)})")
                    # Pad with empty strings if results are fewer than expected
                    batch_results.extend([""] * (len(batch_bboxes) - len(batch_results)))
                
                # Map batch results back to their original indices in the full results array
                for j, idx in enumerate(batch_indices):
                    recognize_results[idx] = batch_results[j] if j < len(batch_results) else ""
            except Exception as e:
                # Handle batch processing errors by logging and setting empty results
                print(f"Batch OCR recognition error: {e}")
                # Set empty results for all items in the failed batch
                for idx in batch_indices:
                    recognize_results[idx] = ""

    def _process_hierarchical_layout(self, 
                                   image: Image.Image, 
                                   layout_items: List[List]) -> tuple:
        """
        Process hierarchical layout relationships among layout items.
        
        Converts bbox_2d to string format for inference in the following format:
        <x_211><y_189><x_362><y_245><LAYOUT_HEADER><000>\n<x_675><y_173><x_1041><y_267><LAYOUT_HEADER><001>
        
        Args:
            image: Input PIL Image
            layout_items: List of layout items, each containing [x1, y1, x2, y2, layout_type]
            
        Returns:
            Tuple of (response_text, new_to_orig_idx):
            - response_text: Hierarchical relationship string from model
            - new_to_orig_idx: Mapping from new index (str) to original index (int)
        """
        if not layout_items:
            return '', {}
        
        try:
            # Build hierarchical relationship prompt
            formatted_bbox_strings = []
            # Mapping from new index to original index: new_index -> original_idx
            index_mapping_dict = {}
            
            for item_index, layout_item in enumerate(layout_items):
                x1, y1, x2, y2, layout_type = layout_item
                
                # Standardize label format
                if not layout_type.startswith("<"):
                    layout_type = f"<LAYOUT_{layout_type.upper()}>"
                
                formatted_bbox_string = f'<x_{x1}><y_{y1}><x_{x2}><y_{y2}{layout_type}<{item_index:03d}>'
                formatted_bbox_strings.append(formatted_bbox_string)
                
                # Record mapping: new index -> original index
                index_mapping_dict[f"{item_index:03d}"] = item_index
            
            if not formatted_bbox_strings:
                return '', {}
            
            # Combine all bbox strings
            combined_layout_string = "\n".join(formatted_bbox_strings)
            
            # Create hierarchical analysis query
            hierarchy_query = combined_layout_string + PROMPT_DICT['hierarchy_recognize']
            
            # Perform hierarchical relationship inference
            hierarchy_response = self._inference(image, hierarchy_query)
            print(f"Hierarchical layout response: {hierarchy_response}")
            
            return hierarchy_response, index_mapping_dict
            
        except Exception as e:
            print(f"HIERARCHICAL layout processing error: {e}")
            return '', {}

    def _parse_single_image(self,
                           image: Image.Image, 
                           batch_size: int = 5) -> tuple:
        """
        Parse a single image and extract layout and content.
        
        Args:
            image: Input PIL Image
            batch_size: Batch size for OCR processing
            
        Returns:
            Tuple of (result, hierarchy_json):
            - result: List of parsed layout items with content
            - hierarchy_json: Hierarchical structure JSON
        """
        # Step 1: Layout detection
        prompt_layout = self._get_prompt("layout_detect")
        response_layout = self._inference(image, prompt_layout)
        print(f"Layout detection response: {response_layout}")
        
        layout_items = parse_layout_str(response_layout)
        if not layout_items:
            return [], {"total_nodes": 0, "root_count": 0, "max_level": 0, "nodes": []}
        
        # Step 2: Categorize layout items by processing type
        categories = categorize_layout_items(layout_items)
        
        # Initialize results storage
        recognize_results = [""] * len(layout_items)
        
        # Step 3: Process different types of layout items
        self._process_figure_items(
            image, layout_items, categories['figure_indices'], recognize_results
        )
        
        self._process_chart_items(
            image, layout_items, categories['chart_indices'], recognize_results
        )
        
        self._process_table_items(
            image, layout_items, categories['table_indices'], recognize_results
        )
        
        self._process_ocr_items_batch(
            image, layout_items, categories['ocr_indices'], 
            recognize_results, batch_size
        )
        
        # Step 3.5: Process hierarchical layout relationships
        hierarchy_results, new_to_orig_idx = self._process_hierarchical_layout(image, layout_items)
        
        # Step 4: Build final results
        result = []
        for i, layout_item in enumerate(layout_items):
            x1, y1, x2, y2, layout_type = layout_item
            
            item = {
                "type": layout_type,
                "bbox": [int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2)]
            }
            if recognize_results[i]:
                item["content"] = recognize_results[i]
            
            result.append(item)
        
        # Step 5: Build hierarchy JSON with full information (structure, content, coordinates)
        hierarchy_json = build_hierarchy_json(
            hierarchy_results, result, new_to_orig_idx
        )
        
        return result, hierarchy_json

    def parse_pdf(self,
                  input_path: str, 
                  dpi: int = 200, 
                  start_page_idx: int = 0, 
                  end_page_idx: int = -1) -> List[Dict]:
        """Parse PDF file by converting to images and processing each page."""
        filename, ext = os.path.splitext(os.path.basename(input_path))
        # Convert PDF pages to images
        images = load_images_from_pdf(
            pdf_path=input_path,
            dpi=dpi,
            start_page_idx=start_page_idx,
            end_page_idx=end_page_idx
        )

        result_pdf = []
        # Process each page
        for i, image in enumerate(images):
            page_result, hierarchy_json = self._parse_single_image(image)
            result_page = {
                "filename": filename,
                "page_idx": i + start_page_idx,
                "image": image,
                "page_result": page_result,
                "hierarchy_json": hierarchy_json,
            }
            result_pdf.append(result_page)
        
        return result_pdf

    def parse_image(self, input_path: str) -> List[Dict]:
        filename = os.path.splitext(os.path.basename(input_path))[0]
        image = load_image(input_path)
        page_result, hierarchy_json = self._parse_single_image(image)

        return [{
            "filename": filename,
            "image": image,
            "page_result": page_result,
            "hierarchy_json": hierarchy_json,
        }]

    def _generate_output_path(self, input_path: str, output_dir: str) -> str:
        json_path = os.path.splitext(os.path.basename(input_path))[0] + '.json'
        json_path = os.path.join(output_dir, json_path)
        
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        return json_path

    def parse_file(self,
                   input_path: str, 
                   output_dir: str, 
                   dpi: int = 200, 
                   start_page_idx: int = 0, 
                   end_page_idx: int = -1) -> None:
        """Parse file and save results in multiple formats (JSON, MD, PNG)."""
        base_filename, file_extension = os.path.splitext(os.path.basename(input_path))

        # Dispatch to appropriate parser based on file extension
        if file_extension in PDF_EXT:
            # Parse PDF file with specified DPI and page range
            parsing_results = self.parse_pdf(
                input_path, dpi=dpi, 
                start_page_idx=start_page_idx, end_page_idx=end_page_idx
            )
        elif file_extension in IMAGE_EXT:
            # Parse single image file
            parsing_results = self.parse_image(input_path)
        else:
            # Unsupported file format
            raise ValueError(f"Input file extension {file_extension} not supported.")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save parsing results for each page in multiple formats
        for page_result in parsing_results:
            # Generate base output file path
            base_output_path = self._generate_output_path(input_path, output_dir)
            
            # Save structured data as JSON format
            with open(base_output_path, "w", encoding='utf-8') as json_file:
                json.dump(page_result["page_result"], json_file, ensure_ascii=False, indent=2)
            
            # Save hierarchical structure as separate JSON file
            hierarchy_output_path = os.path.splitext(base_output_path)[0] + "_hierarchy.json"
            with open(hierarchy_output_path, "w", encoding='utf-8') as hierarchy_file:
                json.dump(page_result["hierarchy_json"], hierarchy_file, ensure_ascii=False, indent=2)
            
            # Save readable text content as Markdown format
            markdown_output_path = os.path.splitext(base_output_path)[0] + ".md"
            with open(markdown_output_path, "w", encoding='utf-8') as markdown_file:
                content_list = [item["content"] for item in page_result["page_result"] if "content" in item]
                markdown_file.write("\n\n".join(content_list))
            
            # Save layout visualization as PNG image
            layout_image = draw_layout_on_image(page_result['image'], page_result['page_result'])
            visualization_output_path = os.path.splitext(base_output_path)[0] + "_layout.png"
            layout_image.save(visualization_output_path)

    def parse_list(self, list_path: str, output_dir: str) -> None:
        # Read file list from text file, filtering out empty lines
        with open(list_path, 'r', encoding='utf-8') as f:
            file_list = [line.strip() for line in f.readlines() if line.strip()]
        
        # Process each file sequentially with progress tracking
        for file_path in tqdm(file_list, desc="Processing files"):
            try:
                # Parse individual file and save results to output directory
                self.parse_file(file_path, output_dir)
            except Exception as e:
                # Log error and continue processing remaining files
                print(f"Error processing {file_path}: {e}")

def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Youtu OCR Parser Client")

    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8080/v1",
        help="Server OpenAI-compatible API URL (should end with /v1)"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="youtuvl",
        help="Model name to use for API calls"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input image or PDF file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF to image conversion"
    )
    
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Initialize OCR parser client
    ocr_parser_client = YoutuOCRParserClient(
        base_url=args.base_url,
        model_name=args.model_name
    )
    
    # Parse the specified file
    ocr_parser_client.parse_file(
        input_path=args.input_path,
        output_dir=args.output_dir,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
