### 2023_Haghighi_IJCAI


## A. Benchmark In-situ sequencing resource
### A.1. Image dataset

We created a benchmark dataset of cells treated with a library of genetic reagents tagged with 186 barcodes of nine digits each. We provide two plate of cells consists of two rows and three columns of wells. Each well contains about 500000 single cells and 20 million four-channel spots in ISS images of nine cycles. Not all cells have spots, and some cells have more than one spot. Each well is imaged at 316 sites (locations) within the well; we used our in-house pipelines to correct for general microscopic illumination patterns on each image and align the images across cycles in order to correct for small differences in physical plate position on the microscope for each cycle’s imaging. The images are subsequently stitched, scaled, and recropped into 100 larger "pseudo-site" images. Each pseudo-site’s image dimensions are (x:5500, y:5500, channels:4, cycles:9). 

This preprocessed dataset available at cell-painting-gallery s3 bucket as a bench-marking resource for developing computational barcode calling methods using ISS images.
- [Link to the data in Gallery](x)

### A.2. Validation resource for cell-level barcode abundance in a pool of single cells.
As there is no direct ground truth for the barcodes assigned to each image location, we evaluate the barcode calling performance in an indirect way. By applying Next-Generation-Sequencing (NGS) on the pooled screens we can quantify the expressed integrated barcodes. We applied NGS to a separate sample of the same cell population that was placed into the 6 wells of our plate, enabling us to count the number of cells perturbed by each barcode. Abundance of transcripts for these genomically integrated barcodes were captured by kallisto tool. Abundance of barcodes based on any decoding strategy applied on this dataset can be calculated as a post-processing step. We can then compare barcode calling strategies to the NGS abundance measures as the experiments’ ``perturbation abundance ground-truth".
The NGS data is at the bulk level, i.e. a pool of cells are all sequenced together. Because each barcode integrates into the cell’s DNA once and only once, and most cells receive only a single integration due to our experimental setup, the NGS information approximates the abundance of cells with a specific transcript or barcode. By contrast, the image-based barcode calling methods read out mRNA spots which can be present in variable copy numbers per cell rather than genomic DNA (which is present in only one copy per cell). For this reason, we cannot expect the number of NGS reads of barcoded cells to linearly correlate with the number of image-based reads of barcoded mRNA transcripts; to assess correlation of our results with NGS data we therefore first need to assign barcode spots to cells to produce cell-level barcode assignments. 
As explained in Section 4.5 of the paper, for the methods which provide a confidence metric on detected barcodes, we assign each cell with the most confident barcode within that cell. For the methods with no confidence scores on the detected barcodes, we assign the barcode with the largest number of occurrence to each cell. And in the case of multiple barcodes with equal occurrence rate, we simply skip the cell assignment.
- [NGS data](x)


### A.3. Evaluation Metrics.
We aim to achieve the highest possible number of cells with a correct barcode assignment. Therefore, the main evaluation metrics are rate of cell recovery and the matches between abundance of cell assignments and the NGS-based barcode abundance:

- ### Cell Recovery Rate. 
  - Recovery rate is defined as the ratio of cell assignments with a targeted barcode over the total number of detected cells by CellProfiler. Note that there are a number of cells that dont recieve any barcode assignments and therefore this number is different than the PPV at the cell level.

- ### NGS match. 
As described in Section A.2, NGS-based relative abundance of each barcode in the experiment serves as an indirect ground truth to assess the quality of the relative abundance of the detected barcodes assigned to the cells which also exist in the experimental codebook. The similarity between the abundance of the detected cell-level barcode assignments and the NGS-based abundance of codebook barcodes is measured by $R^2$ between the two abundance distributions.

- ### False Discovery Analysis. 
The next set of metrics are calculated for false discovery analysis. Following figure illustrates the distinction among the various types of barcode assignments used in the evaluation metrics. The codebook inputted to the decoding algorithms contains two sets of barcodes:
- **Targeted** barcodes which form the experimental codebook or the experiments reference library of barcodes.
- **Trick** barcodes which are a set of artifically generated barcodes that are not in the experimental codebook but are faked and can be used for assessing the overfitting issues a decoding algorithm may have.
![](./documentation/images/codebooks2.png)

<!-- - ### Assignment Rates.  -->
    
    - $PPV$ Correct assignment rate refers to the ratio of the calls which are in the targeted list and serves as a quality metric for a barcode calling method. We report this metric as the Positive Predictive Value (PPV) at each barcode and cell level detections.

    - $FDR$ Incorrect assignment rates for two categories of "trick" and "not-targeted-nor-trick" calls are reported as False Discovery Rates (FDR) and are denoted as $FDR_{trick}$ and $FDR_{other}$ respectively.
- Barcode assignments are evaluated at both spot-level and cell-level 
Evaluation metrics are based on the cell calling assignments. 






 


