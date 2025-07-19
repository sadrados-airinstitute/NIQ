from api import API
import uvicorn

def main():
    # Create an instance of the API class
    api_instance = API()

    # Set up the API endpoints
    api_instance.create_endpoints()

    # Run the FastAPI app using Uvicorn
    # You can customize the host and port as needed
    uvicorn.run(api_instance.app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Run the main function to start the app
    main()