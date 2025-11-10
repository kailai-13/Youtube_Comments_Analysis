            # Youtube_Comments_Analysis
It's A ACalled Youtube Analysis And Hate Speech Detection.
      
As In Name It Have Two Pages One Is Sentiment Analysis And Another Is Hate Speech Analysis
    
   
It Only Requires A YouTube Video Link That You Want To Analyze And Then It Runs The Transformer Models On The Comments Of That Video. It Gives Graph And Analysis Of Comment's Emotions With Abusive Language Analysis.
  
NOTE:It Uses Models That Are Configured For Tamil And English Language

To Run It Follow The Following Steps. 
  
1. Clone the Repository:

        git clone  https://github.com/kailai-13/Youtube_Comments_Analysis/
        cd Youtube_Comments_Analysis


3. Set Up a Virtual Environment (Optional but Recommended):
Create and activate a virtual environment to isolate dependencies.

        python -m venv env
        env\Scripts\activate

3. Install Dependencies:
      Install all required Python packages specified in the requirements.txt file.

        pip install -r requirements.txt


4. Set Up the Database:
      Run Django migrations to set up the database.

        python manage.py makemigrations
        python manage.py migrate

5. Run the Development Server:
      Start the Django development server to test the app locally.

        python manage.py runserver
      Open your browser and navigate to http://127.0.0.1:8000/ to view your app.



thank you if you see this 
