
def convertToJpg(cwd):
    ### This function accepts single parameter ie. directory path and
    # convert all pdf and other images format in jpg 
    ###
    # import module
    from pdf2image import convert_from_path
    ## check file type from any directory
    import os
    from os import listdir
    from os.path import isfile, join
    from PIL import Image
    # cwd = os.getcwd()
    # cwd = '/Users/neerajyadav/Desktop/DataSet'
    onlyfiles = [os.path.join(cwd, f) for f in os.listdir(cwd) if 
    os.path.isfile(os.path.join(cwd, f))]
    for file in onlyfiles:
        # print(file)
        if os.path.isfile(file):
            file_extension = os.path.splitext(file)[1]
        print(file_extension)
        if file_extension.lower() in {'.mp3', '.flac', '.ogg'}:
            print("It's a music file")
        elif file_extension.lower() in {'.png'}:
            # print("It's an image file")
            print(file)
            im1 = Image.open(file).convert('RGB')
            # filen  = file.replace('.jpeg','')
            filen = file.replace('.png','')
            # print('fn',filen)
            im1.save(filen+'.jpg')
            pass
        elif file_extension.lower() in {'.pdf'}:
            images = convert_from_path(file)
            # print("Its a pdf file")
            for i in range(len(images)):
                    filen = file.replace('.pdf','')
                    images[i].save(filen+'.jpg', 'JPEG')
                    print('converted')
        else:
            print('Unsupported format files.')

    ## pass a file to directly 