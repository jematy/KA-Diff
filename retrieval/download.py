import img2dataset

def main():
    input_file = "dataset_wit.tsv" 
    output_folder = "output_images"

    img2dataset.download(
        url_list=input_file,             
        output_folder=output_folder,       
        input_format="tsv",                
        url_col="image_url",                
        caption_col="caption_reference_description",
        image_size=None,                  
        retries=4,                         
        thread_count=64,                   
        resize_mode=None                   
    )

if __name__ == "__main__":
    main()
