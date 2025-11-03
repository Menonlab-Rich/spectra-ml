import os
import numpy as np
import tifffile
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--output", type=str, default="tiffs", help="Output Directory")
parser.add_argument("tiff", type=str, help="Path to tiff file to export.")

args = parser.parse_args()

def export_tiff_pages():
    """
    Scans a source directory for TIFF files and saves each page of a
    multipage TIFF as a separate TIFF file in the output directory.
    """
    # Create the output directory if it doesn't already exist
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory is: {args.output}")

    try:
        # Open the multipage TIFF file using tifffile
        # The 'with' statement ensures the file is properly closed
        with tifffile.TiffFile(args.tiff) as tif:
            # Get the base name of the file to use in the output filenames
            base_filename = Path(args.tiff).stem
            
            # Iterate through each page in the TIFF file
            # enumerate provides both the index (page_num) and the page object
            for page_num, page in enumerate(tif.pages):
                # Convert the page data to a NumPy array
                page_data = page.asarray()

                if np.max(page_data) == 0:
                    continue
                # Construct the output filename
                # We pad the page number with zeros (e.g., 001, 002) for proper sorting
                output_filename = f"{base_filename}_page_{page_num + 1:03d}.tif"
                output_path = os.path.join(args.output, output_filename)

                # Save the NumPy array as a new TIFF file
                tifffile.imwrite(output_path, page_data)
                print(f"  - Saved page {page_num + 1} as {output_filename}")

    except Exception as e:
        print(f"  - Could not open or process {args.tiff}. Error: {e}")
        return

    print("\n--- Script finished. ---")

if __name__ == "__main__":
    export_tiff_pages()
