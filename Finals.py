import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


st.set_page_config(page_title='Tips Prediction using Linear Regression', layout='wide')



@st.cache_data
def load_data():
    df = pd.read_csv('tip.csv')
    return df

tips = load_data()


st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B0C10; /* Dark background for the entire app */
    }
    
    body {
        font-family: 'Arial', sans-serif;
        background-color: #0B0C10; /* Dark background for the entire app */
        color: #ffffff; /* Light text color */
    }
  
    h1, h2, h3, h4 {
        color: #ffffff; /* Light headers */
    }
    
    .sidebar .sidebar-content {
        background-color: #2c2c2c; /* Dark sidebar */
        color: #ffffff; /* Light text in sidebar */
    }

    .css-1aumxhk {
        background-color: #222; /* Dark background for containers */
    }

    .rounded-border {
        border-radius: 15px;
        border: 1px solid #444; /* Slightly lighter border */
        padding: 10px;
        margin-bottom: 20px;
        background-color: #2b2b2b; /* Dark background for cards */
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    
    .stButton>button {
        width: 100%;
        font-weight: bold;
        background: linear-gradient(90deg, #45A29E, #4e69ba); /* Facebook blue gradient */
        color: white; /* Text color */
        border-radius: 8px;
        height: 50px;
        border: none;
        transition: background 0.3s; /* Smooth transition */
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #66FCF1, #3d5a99); /* Darker gradient on hover */
        color: #fff; /* Ensure text stays white */
    }

    .stButton>button:active {
        background: linear-gradient(90deg, #66FCF1, #3d5a99); /* Keep the hover state on press */
        color: #fff; /* Keep text color white */
    }

    .stButton>button:focus {
        outline: none;
        border: none;
    }
    
    .st-expander {
        background-color: #2c2c2c; /* Dark background for expanders */
        border: 1px solid #444; /* Darker border */
        border-radius: 5px;
        padding: 15px;
    }
    
    .st-dataframe {
        background-color: #222; /* Dark background for dataframes */
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True
)

nav_col1, main_col, nav_col2 = st.columns([1, 5, 1]) 

# Initialize selected_tab in session state if not already set
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Home"


with nav_col1:
    st.write("")

with nav_col2:
    st.write("")
    

with main_col:

    st.markdown(
        """
        <div style="background-color: #1F2833; padding: 10px; border-radius: 10px;">
            <h1 style="text-align: center; color: #;">💸 Tips Prediction using Linear Regression</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .stButton>button {
            width: 100%;
            font-weight: 900;
            height: 50px; /* You can adjust this height as needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4, tab5 = st.columns(5)

 
    with tab1:
        if st.button("📖 Introduction", key="tab_home"):
            selected_tab = "Home"
    with tab2:
        if st.button("📊 Univariate Analysis", key="tab_count"):
            selected_tab = "Univariate Analysis"

    with tab3:
        if st.button("📈 Multivariate Analysis", key="tab_statistics"):
            selected_tab = "Multivariate Analysis"

    with tab4:
        if st.button("🔎 Linear Regression", key="tab_laptop_features"):
            selected_tab = "Linear Regression"

    with tab5:
        if st.button("🔚 Results", key="tab_multivarient_analysis"):
            selected_tab = "Results"

    st.markdown("<hr style='border: 1px solid #444; margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "Home"

    if 'selected_tab' in locals() and selected_tab:
        st.session_state.selected_tab = selected_tab

    font_size = 12  
    title_size = 16  


    if st.session_state.selected_tab == "Home":

        with st.container():
            col1, col2 = st.columns([3, 2])

            with col1:
                st.subheader('About the Data:')
                st.write('The **tips** dataset is a popular dataset often used for demonstration and practice in data analysis and visualization. It contains information about various attributes of customers in a restaurant, including the **total bill amount** (the cost of food and drinks), the **tip amount** given by the customer, the **gender** of the customer (e.g., Male or Female), whether the customer is a **smoker** or not (e.g., Yes or No), the **day** of the week when the transaction occurred (e.g., Sun, Sat, Thu, etc.), the **time** of day when the transaction occurred (typically categorized as Lunch or Dinner), and the **size** of the party or group of customers.')

                st.write('In this analysis, our goal is to explore the relationships between these attributes and predict the tip amount using **linear regression**. By examining how factors such as the total bill, time of day, and the size of the party influence the tip, we aim to build a model that can accurately predict tip amounts based on these attributes.')

            with col2:
                with st.expander("Data Preview", expanded=True):
                    st.dataframe(tips.head())




    elif st.session_state.selected_tab == "Univariate Analysis": 
        
    # st.subheader('Data Overview')
        nominal_columns = ["sex", "smoker", "day", "time"]

        # Color palettes for each nominal column
        color_palettes = {
            "sex": px.colors.qualitative.Safe,
            "smoker": px.colors.qualitative.Set2,
            "day": px.colors.qualitative.Set3,
            "time": px.colors.qualitative.Pastel
        }

        # Create columns for displaying the pie charts in a row
        columns = st.columns(len(nominal_columns))

        # Populate each column with a pie chart
        for col, column in zip(columns, nominal_columns):
            with col:
                fig = px.pie(
                    tips,
                    names=column,
                    title=f"{column.capitalize()} Distribution",
                    color_discrete_sequence=color_palettes[column],  # Use unique color palette
                    hole=0.3  # Optional, for a donut chart style
                )
                
                # Adjust text size and legend position
                fig.update_layout(
                    title_font_size=24,  # Make the title bigger
                    title_x=0.5,  # Center the title horizontally within the plot area
                    title_y=0.95,  # Slightly adjust vertical position
                    title_xanchor="center",  # Ensure title is anchored to center
                    title_yanchor="top",  # Ensure title is anchored to the top of the chart
                    legend=dict(
                        orientation="h",  # Horizontal legend
                        yanchor="bottom",  
                        y=-0.2,  
                        xanchor="center",  
                        x=0.5,
                        font=dict(
                            size=16,  # Increase legend font size
                        ) 
                    ),
                    font=dict(
                        size=20,  # Make the text inside the pie chart bigger
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add concise description for each pie chart
                if column == "sex":
                    with st.expander("Observation",expanded=True):
                        st.markdown("""
                            - Majority of diners are male (157), with a smaller proportion of females.
                        """)
                elif column == "smoker":
                    with st.expander("Observation",expanded=True):
                        st.markdown("""
                            - More diners are non-smokers (151) than smokers (93).
                        """)
                elif column == "day":
                    with st.expander("Observation",expanded=True):
                        st.markdown("""
                            - Saturday is the most popular dining day (87 observations).
                        """)
                elif column == "time":
                    with st.expander("Observation",expanded=True):
                        st.markdown("""
                            - Dinner (176 observations) is more popular than lunch.
                        """)

        # Non-nominal columns for histograms
        non_nominal_columns = ["tip", "total_bill", "size"]

        st.markdown("---")

        # st.subheader("Frequency Distribution of Non-Nominal Values")

        # Create columns for displaying histograms side by side
        columns = st.columns(len(non_nominal_columns))

        # Loop through non-nominal columns and display histograms
        for col, column in zip(columns, non_nominal_columns):
            with col:
                # Create the histogram using Plotly
                fig = px.histogram(
                    tips,
                    x=column,
                    nbins=10,  # Number of bins in the histogram
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                # Adjust layout
                fig.update_layout(
                    title=f"{column.replace('_', ' ').capitalize()} Distribution",
                    title_font_size=24,  # Make the title bigger
                    title_x=0.5,  # Center the title horizontally within the plot area
                    title_y=0.95,  # Slightly adjust vertical position
                    title_xanchor="center",  # Ensure title is anchored to center
                    title_yanchor="top",  # Ensure title is anchored to the top of the chart
                    xaxis_title=None,  # Remove x-axis title
                    yaxis_title="Frequency",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add concise description for each histogram
                if column == "tip":
                    with st.expander("Observation",expanded=True):
                        st.markdown("""
                            - Tips range from 1.00 to 10.00, concentrating around 2.00.
                        """)
                elif column == "total_bill":
                    with st.expander("Observation",expanded=True):
                        st.markdown("""
                            - Bills range from 3.07 to 50.81, with an average of 19.79.
                        """)
                elif column == "size":
                    with st.expander("Observation",expanded=True):
                        st.markdown("""
                            - The typical party size is small, with most parties of 2 or 3.
                        """)
       


    elif st.session_state.selected_tab == "Multivariate Analysis":
            
        tips = px.data.tips()

            # Ensure 'sex' is treated as a categorical variable
        tips['sex'] = pd.Categorical(tips['sex'])

            # Create scatter plot with independent colors for 'sex'
        fig6 = px.scatter(
                tips,
                x='total_bill',  # x-axis: total_bill
                y='tip',  # y-axis: tip
                size='size',  # Marker size: size
                color='sex',  # Marker color based on sex
                color_discrete_sequence=px.colors.qualitative.Pastel,  # Use a discrete color sequence
                symbol='time',  # Marker style based on time
                opacity=0.7,  # Transparency of markers
                title="Total Bill vs Tip",
                labels={"total_bill": "Total Bill", "tip": "Tip", "size": "Size", "sex": "Sex", "time": "Time"}
            )

        fig6.update_layout(
                title_font_size=20,  # Make the title bigger
            )

            # Show the plot in Streamlit
        st.plotly_chart(fig6, use_container_width=True)


        # Expander for detailed analysis
        with st.expander("Key Takeaways"):
            st.write("**1. Positive Correlation**:")
            st.write("""
            There is a noticeable trend where higher total bills tend to result in larger tip amounts. This suggests that customers generally tip in proportion to the cost of their meal.
            """)

            st.write("**2. Gender Differences**:")
            st.write("""
            There is a slight difference in tipping patterns between men and women. On average, men tend to leave slightly larger tips compared to women.
            """)

            st.write("**3. Smoking Status**:")
            st.write("""
            The presence of smoking does not seem to significantly affect tipping behavior. No strong correlation between smoking and tip amounts is observed.
            """)

            st.write("**4. Day and Time Effects**:")
            st.write("""
            Tipping habits change depending on the day of the week and time of day. Generally, customers tend to tip more on weekends and during dinner hours, which could be due to higher restaurant traffic at those times.
            """)



        tabs = st.tabs(["Tips", "Total Bill", "Correlation"])

        with tabs[0]:
    
            col1, col2 = st.columns(2)
            with col1:
               
                fig1 = px.box(
                    tips,
                    x='sex',  
                    y='tip',  # y-axis: tip
                    color='smoker',  # Color based on smoker status
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    title="  Tips by Gender and Smoker Status",
                    labels={"sex": "Gender", "tip": "Tip", "smoker": "Smoker Status"}
                )

                # Remove the x-axis title and move the legend to the bottom
                fig1.update_layout(
                    xaxis_title=None,  # Remove x-axis title
                    legend_title=None,
                    title_font_size=20,  # Make the title bigger

                    legend=dict(
                        orientation="h",  # Set legend to horizontal
                        yanchor="bottom",  # Set the anchor for the legend at the bottom
                        y=-0.2,  # Position the legend below the plot
                        xanchor="center",  # Center the legend horizontally
                        x=0.5  # Position the legend in the center
                    )
                )

                st.plotly_chart(fig1, use_container_width=True)

            # Bar Plot in the second column
            with col2:
                # Group data by day and calculate average tip amount
                avg_tip_by_day = tips.groupby('day')['tip'].mean().reset_index()
                # Sort the DataFrame by day
                avg_tip_by_day = avg_tip_by_day.sort_values(by='day')

                # Create the bar plot using Plotly
                fig2 = px.bar(
                    avg_tip_by_day,  # Data
                    x='day',  # x-axis: day
                    y='tip',  # y-axis: average tip
                    title="  Total Tip Amount by Day",
                    color='tip',  # Color bars based on tip value
                    color_continuous_scale='blues',  # Color scale
                    labels={'day': 'Day of the Week', 'tip': 'Average Tip Amount'}
                )
                fig2.update_layout(
                    xaxis_title=None,  # Remove x-axis title
                    title_font_size=20,  # Make the title bigger
                )
                
                st.plotly_chart(fig2, use_container_width=True)

        with tabs[1]:
            col1, col2 = st.columns(2)
            tips = px.data.tips()

            # Ensure 'sex' is treated as a categorical variable
            tips['sex'] = pd.Categorical(tips['sex'])
            with col1:
                total_bill_by_day_gender = tips.groupby(['day', 'sex'])['total_bill'].sum().reset_index()

                # Create the plot
            # Create the plot
                fig3 = px.bar(total_bill_by_day_gender, 
                            x='day', 
                            y='total_bill', 
                            color='sex', 
                            title="  Total Bill Amount by Day and Gender",
                            color_discrete_sequence=px.colors.qualitative.Safe)

                # Update the layout to display bars side by side
                fig3.update_layout(
                    title_font_size=20,
                    barmode='group',
                    legend_title=None,  # Set the legend title to 'Sex'
                        legend=dict(
                        orientation='h',  # Horizontal layout for the legend
                        yanchor='bottom',  # Anchor the legend at the bottom
                        y=-0.2,  # Position it slightly below the plot area
                        xanchor='center',  # Center the legend horizontally
                        x=0.5  # Place the legend in the middle
                    )
                )

                # Update the axis labels
                fig3.update_xaxes(title=None)  # Set x-axis label to None
                fig3.update_yaxes(title='Total Bill')  # Set y-axis label to 'Total Bill'

                # Show the plot
                st.plotly_chart(fig3, use_container_width=True)
            with col2:
                # Pivot the data as done in the original code
                pivot_tip = tips.pivot_table(index='day', columns='time', values='total_bill')

                # Create the heatmap using Plotly
                fig4 = go.Figure(data=go.Heatmap(
                    z=pivot_tip.values,  # The values of the heatmap
                    x=pivot_tip.columns,  # The columns (time)
                    y=pivot_tip.index,  # The index (day)
                    colorscale='Blues', # Color scale
                    colorbar=dict(title="Total Bill"),  # Color bar title
                    text=pivot_tip.round(2).values,  # Annotate with the values rounded to 2 decimals
                    hovertemplate="%{text}",  # Display values when hovering
                ))

                # Set the title
                fig4.update_layout(
                    title='  Total Bill Amount by Day and Time',
                    title_font=dict(size=20),  # Correct way to set title font size
                    xaxis_title='Time',
                    yaxis_title='Day'
                )

                # Show the plot
                st.plotly_chart(fig4, use_container_width=True)

        with tabs[2]:

            label_encoder = LabelEncoder()
            tips["sex"] = label_encoder.fit_transform(tips["sex"])
            tips["smoker"] = label_encoder.fit_transform(tips["smoker"])
            tips["day"] = label_encoder.fit_transform(tips["day"])
            tips["time"] = label_encoder.fit_transform(tips["time"])

                # Correlation matrix
            corr = tips.select_dtypes(include=[np.number]).corr()

                # Create heatmap using Plotly
            fig5 = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale='Blues',  # Using Plotly's Blues colorscale
                    zmin=-1, zmax=1,
                    colorbar=dict(title="Correlation"),
                    hoverongaps=False
                ))

            fig5.update_layout(
                    title='Correlation Heatmap',
                    xaxis=dict(title='Features'),
                    yaxis=dict(title='Features')
                )

                # Show the plot
            st.plotly_chart(fig5, use_container_width=True)
            with st.expander("🔍 Insights"):
            
                    st.write(
                            """
                            - **Total Bill**: The total bill has the highest correlation (0.68) with tips. 
                            It's no surprise that the more you spend, the more you tend to tip!
                            - **Size**: The number of people in the party also has a decent correlation (0.49) with tips, 
                            as larger parties often leave larger tips.
                            - **Time**: There’s a moderate correlation (0.12) with the time of day. This might suggest 
                            slight variations in tips between lunch and dinner hours.
                            - **Sex**: The gender of the customer has a low correlation (0.09) with tips, indicating 
                            that tip amounts aren’t strongly influenced by gender.
                            - **Day**: The day of the week has an almost negligible correlation (0.01), which implies 
                            that tips don’t significantly change based on the day of the week.
                            - **Smoker**: The smoking status shows an extremely low correlation (0.0059), meaning 
                            smoking habits have minimal to no impact on tip amounts.
                            """
                        )

                    st.write(
                            "### Preparing The Data"
                        )
                    st.write(
                            "The dataset is already clean, with no null values or other issues. However, some columns, such as 'sex', 'smoker', 'day', and 'time', contain nominal (categorical) data. "
                            "Before we can proceed with calculating correlations and performing linear regression, we need to convert these categorical variables into numerical values. "
                            "LabelEncoder is used for this task, as it transforms categorical labels into a numerical format, which allows us to properly assess the relationships between the variables and the tip amount."
                        )





  
    elif st.session_state.selected_tab == "Linear Regression": 
        
        # Load the tips dataset from seaborn
        tips = sns.load_dataset('tips')

        # Define the features and target variable
        X = tips[['total_bill', 'size']]  # independent variables
        y = tips['tip']  # dependent variable

        # Create and fit the regression model
        model = LinearRegression()
        model.fit(X, y)

        # Get the predicted values
        y_pred = model.predict(X)

        # Prepare the meshgrid for the regression plane
        x_range = np.linspace(tips['total_bill'].min(), tips['total_bill'].max(), 30)
        y_range = np.linspace(tips['size'].min(), tips['size'].max(), 30)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)

        # Predict the z values (tips) for each point on the grid
        z_mesh = model.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])
        z_mesh = z_mesh.reshape(x_mesh.shape)

        # Create the 3D scatter plot and surface plot using Plotly
        fig = go.Figure()

        # Add the scatter plot of actual data
        fig.add_trace(go.Scatter3d(
            x=tips['total_bill'],
            y=tips['size'],
            z=y,
            mode='markers',
            marker=dict(color='blue', size=5, opacity=0.9),
            name='Actual Data'
        ))

        # Add the regression surface plot
        fig.add_trace(go.Surface(
            x=x_mesh,
            y=y_mesh,
            z=z_mesh,
            colorscale='reds',
            opacity=0.5,
            name='Regression Plane'
        ))

        # Labels and title
        fig.update_layout(
            scene=dict(
                xaxis_title='Total Bill',
                yaxis_title='Size',
                zaxis_title='Tip'
            ),
            title='  3D Regression Plot: Tip vs. Total Bill and Size',
            scene_camera=dict(
                eye=dict(x=1.25, y=1.25, z=1)
            ),
            width=1200,  # Increase the width
            height=800,  # Increase the height
        )

        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
        

        

    elif st.session_state.selected_tab == "Results":
        col1, col2 = st.columns(2)
 
        with col1:
            label_encoder = LabelEncoder()
            tips["sex"] = label_encoder.fit_transform(tips["sex"])
            tips["smoker"] = label_encoder.fit_transform(tips["smoker"])
            tips["day"] = label_encoder.fit_transform(tips["day"])
            tips["time"] = label_encoder.fit_transform(tips["time"])

            # Prepare the data
            X = tips[['total_bill', 'tip', 'sex', 'smoker', 'day', 'time']]
            y = tips['tip']

            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Print metrics
            print('MAE is : ',mae)
            print('MSE is : ',mse)
            print('R2 is  : ',r2)

            # Sample data
            sample_size = 100
            sample_data = list(zip(y_test[:sample_size], y_pred[:sample_size]))

            print("Sample of Actual vs Predicted Values:")
            for actual, predicted in sample_data:
                print(f"Actual: {actual}, Predicted: {predicted}")

            # Create Plotly scatter plot for Actual vs Predicted
            fig = go.Figure()

            # Scatter plot for Actual vs Predicted
            fig.add_trace(go.Scatter(
                x=y_test, 
                y=y_pred, 
                mode='markers', 
                marker=dict(color='white', size=8, opacity=0.7),
                name='Actual vs Predicted'
            ))

            # Add a line for perfect prediction (y = x)
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction'
            ))

            # Update layout
            fig.update_layout(
                title='  Actual vs Predicted Values',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)
        with col2:
            residuals = y_test - y_pred

            # Create the Plotly residuals plot
            fig_residuals = go.Figure()

            # Scatter plot for Residuals vs Predicted Values
            fig_residuals.add_trace(go.Scatter(
                x=y_pred, 
                y=residuals, 
                mode='markers', 
                marker=dict(color='white', size=8, opacity=0.9),
                name='Residuals'
            ))

            # Add a line for zero residuals (y = 0)
            fig_residuals.add_trace(go.Scatter(
                x=[min(y_pred), max(y_pred)],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Zero Residual Line'
            ))

            # Update layout for the residuals plot
            fig_residuals.update_layout(
                title='  Residuals Plot',
                xaxis_title='Predicted Values',
                yaxis_title='Residuals',
                showlegend=True
            )

            # Show the Plotly residuals plot
            st.plotly_chart(fig_residuals, use_container_width=True)
        st.subheader("Results")

        # Expander for detailed results and analysis
        with st.expander("View Detailed Results and Analysis"):
            # Display the metrics
            st.write("**MAE (Mean Absolute Error):** 2.5693732855610765e-15")
            st.write("**MSE (Mean Squared Error):** 1.1278497304365206e-29")
            st.write("**R² (Coefficient of Determination):** 1.0")
            
            # Analysis of the results
            st.write("""
            The **MAE** value is extremely low, suggesting that the model's predictions are, on average, very close to the actual values. This indicates high precision in the predictions.
            
            The **MSE** is also very close to zero, which further supports that the errors in predictions are minimal, and the model is not overfitting or underfitting.
            
            The **R² score** is 1.0, meaning that the model explains 100% of the variance in the target variable (`tip`). This is a perfect result, suggesting the model has excellent explanatory power for the data.
            
            In addition, the **Actual vs Predicted** graph looks promising, showing that the predictions align closely with the true values, indicating a strong model performance.
            
            The **residuals** (the difference between the actual and predicted values) appear to be randomly scattered around zero, confirming that there is no significant bias in the model. The residuals plot suggests that the model does not have any systematic errors and performs well across all levels of the input features.
            """)

        
        
            