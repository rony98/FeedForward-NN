/*
 * Author: Rony Verch
 * File Name: onlineStorageHelper.cs
 * Project Name: Shooter From The Sky
 * Creation Date: January 6, 2015
 * Modified Date: January 20, 2015
 * Description: A helper class to upload, download and checks for files that are stored on a server online
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using com.shephertz.app42.paas.sdk.csharp;
using com.shephertz.app42.paas.sdk.csharp.upload;
using System.IO;
using System.Net;

namespace NeuralNetworkV2
{
    class OnlineStorageHelper
    {
        ////////////////////////////////////////////||\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        /////////////////////////////////////// VARIABLES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        ////////////////////////////////////////////||\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


        //Variables for uploading files online
        private UploadService uploadService;
        private Upload upload;
        private App42Response response;


        ///////////////////////////////////////////||\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        ///////////////////////////////////// CONSTRUCTOR \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        ///////////////////////////////////////////||\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


        //Pre: None
        //Post: The online storage helper object is created
        //Desc: A blank constructor for creating the online storage helper object
        public OnlineStorageHelper()
        {
        }


        ///////////////////////////////////////////||\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        /////////////////////////////////////// METHODS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        ///////////////////////////////////////////||\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


        //Pre: None
        //Post: The online storage helper object is initialized
        //Desc: A method for initializing the online storage helper object
        public void Initialize()
        {
            //Initilizes API for uploading file
            App42API.Initialize("07bc061b3322e11203ce8c88715d8880d1e9d6a3bda2fa03169f672a8825c9d5", "8304f4c591dcac89ba3272e9067b6765f52f1ea83a06eb8571acca1669eb2ba5");
            uploadService = App42API.BuildUploadService();
        }


        //Pre: The file name, the file path, the file type, and the description of the file
        //Post: A boolean for whether the file was updated
        //Desc: A method for updating a file
        public bool UpdateFile(string fileName, string filePath, string fileType, string description)
        {
            //Boolean for whether the file was updated or not on the server
            bool success = false;

            //Tries to update the file
            try
            {
                //If the file was removed
                if (RemoveOldFile(fileName))
                {
                    //Upload new one
                    upload = uploadService.UploadFile(fileName, filePath, fileType, description);
                    success = true;
                }
                //If the file was not removed
                else
                {
                    //The success of updating the file is set to false
                    success = false;
                }
            }
            //If there are any exceptions
            catch
            {
                //The success of updating the file is set to false
                success = false;
            }

            //Returns whether the file was updated
            return success;
        }


        //Pre: The file name
        //Post: A boolean for whether an old version of the file was removed
        //Desc: A method for removing the old version of a file
        public bool RemoveOldFile(string fileName)
        {
            //Boolean for whether the old file was removed
            bool success;

            //Tries to remove the file
            try
            {
                //Tries to remove the file
                response = uploadService.RemoveFileByName(fileName);

                //Stores whether the file was removed or not
                success = response.IsResponseSuccess();
            }
            //If there are any exceptions
            catch (Exception ex)
            {
                //If the exception message contains the text in the brackets
                if (ex.Message.Contains("The file with the name '" + fileName + "' does not exist."))
                {
                    //Set the file to removed because it never existed in the first place
                    success = true;
                }
                //If the exception message was anything else
                else
                {
                    //Sets the file to not removed
                    success = false;
                }
            }

            //Returns whether the file was removed
            return success;
        }


        //Pre: The file path and name
        //Post: A boolean for whether the file was downloaded
        //Desc: A method for downloading a file from a server
        public bool DownloadFile(string filePath, string fileName)
        {
            //If a downloaded version of the file currently exists 
            if (File.Exists(filePath))
            {
                //Deletes the downloaded version of the file
                File.Delete(filePath);
            }

            //Tries to download the file
            try
            {
                //A client is created to download the file using a URL
                using (var client = new WebClient())
                {
                    //Every file with the same name of the file that is being looked for is retrieved
                    upload = uploadService.GetFileByName(fileName);
                    IList<Upload.File> fileList = upload.GetFileList();

                    //Loop for every file in the file list
                    for (int i = 0; i < fileList.Count(); i++)
                    {
                        //The current file is downloaded using it's url and the name it is stored as
                        client.DownloadFile(fileList[i].GetUrl(), fileName);
                    }
                }

                //Returns that the file was downloaded
                return true;
            }
            //If there are any exceptions
            catch
            {
                //Returns that the file was not downloaded
                return false;
            }
        }


        //Pre: The file name
        //Post: A boolean for whether the file exists
        //Desc: A method for checking whether a file exists
        public bool CheckFileExists(string fileName)
        {
            //Every file saved on the server is retrieved from the server (not downloaded)
            Upload tempUpload = uploadService.GetAllFiles();
            IList<Upload.File> fileList = tempUpload.GetFileList();

            //Loop for every file in the file list
            for (int i = 0; i < fileList.Count; i++)
            {
                //If the file that is being looked for exists on teh server
                if (fileList[i].GetName() == fileName)
                {
                    //Return that the file exists
                    return true;
                }
            }

            //Return that the file does not exist
            return false;
        }
    }
}
