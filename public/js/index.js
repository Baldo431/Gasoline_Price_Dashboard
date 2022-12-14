const baseURL = "http://localhost:8080/"

// function to gather all data from the database
async function init(){

    await Promise.all([
        fetch(baseURL+'results/current_pricing').then(response=> response.json()),
        fetch(baseURL+'results/predicted_pricing').then(response=> response.json())
    ]).then(value => {
        buildTable(value[0], value[1]);
    });

    await fetch(baseURL+'results/gas_crude_history')
    .then(response=> response.json())
    .then(data => {
        buildLineChart(data);
    });

    Promise.all([
        fetch(baseURL+'results/tweet_data?' + new URLSearchParams({Polarity: "Negative"})).then(response=> response.json()),
        fetch(baseURL+'results/tweet_data?' + new URLSearchParams({Polarity: "Neutral"})).then(response=> response.json()),
        fetch(baseURL+'results/tweet_data?' + new URLSearchParams({Polarity: "Positive"})).then(response=> response.json())
    ]).then(value => {
        buildSentimentChart({"Positive": value[0].length, "Neutral": value[1].length, "Negative": value[2].length});
        insert_tweets(JSON.parse(value[0]), JSON.parse(value[1]), JSON.parse(value[2]));
    });

}

// Build the table showing today's gas prices.
function buildTable(current_data, future_data) {
    // First, clear out any existing data
    var tbody = d3.select("tbody");
    tbody.html("");

    // Parse data into JSON format
    let myJSON = JSON.parse(current_data)

    //Add row header
    let row = tbody.append("tr");
    let cell = row.append("td");
    cell.text("Current Average");

    let fuels = ["Regular", "MidGrade", "Premium", "Diesel"]

    //Iterate through data and add todays gas prices
    fuels.forEach(x => {
        let cell = row.append("td");
        cell.text(`$${myJSON[x].toFixed(2)}`);
    })
}

// Helper function to isolate the column data we want.
function jsonValueExtract(sample, colName){
    let results = [];
    for( const x in sample[colName]){
        results.push(sample[colName][x]);
    }
    return results;
}

// Build out both line charts.
function buildLineChart(sample) {

    // Parse data into JSON format
    const myJSON = JSON.parse(sample)

    // Extract the date field which will be constant across our traces.
    const gasDates = jsonValueExtract(myJSON, "Date");

    var trace1={
        x:gasDates,
        y:jsonValueExtract(myJSON, "Regular"),
        type: 'scatter',
        name: 'Regular'
    };

    var trace2={
        x:gasDates,
        y:jsonValueExtract(myJSON, "MidGrade"),
        type: 'scatter',
        name: 'Midgrade'
    };

    var trace3={
        x:gasDates,
        y:jsonValueExtract(myJSON, "Premium"),
        type: 'scatter',
        name: 'Premium'
    };

    var trace4={
        x:gasDates,
        y:jsonValueExtract(myJSON, "Diesel"),
        type: 'scatter',
        name: 'Diesel'
    };

    var trace5={
        x:gasDates,
        y:jsonValueExtract(myJSON, "Crude Closing"),
        type: 'scatter',
        name: 'Crude Oil'
    };
    
    // Plot the first chart.
    var lineLayout={
        title: "<b>Daily Avg. Price by Fuel Type</b>",
        xaxis: {title: "Date"},
        yaxis: {title: "Price (USD)"}
    };

    var data = [trace1, trace4];

    Plotly.newPlot('lineChart01', data, lineLayout);

    // Plot the second chart.
    var lineLayout2={
        title: "<b>Daily Avg. Crude Oil Price</b>",
        xaxis: {title: "Date"},
        yaxis: {title: "Price (USD)"}
    };

    var data2 = [trace5];

    Plotly.newPlot('lineChart02', data2, lineLayout2);

}

// Build out the stacked bar chart for sentiment polarity.
function buildSentimentChart(sentiment_dict){

    var json_obj = sentiment_dict;
    var total = json_obj.Negative + json_obj.Neutral + json_obj.Positive;

    var trace1 = {
        x: [(json_obj.Positive/total)*100],
        y: [''],
        name: 'Positive',
        orientation: 'h',
        type: 'bar',
        marker:{color: 'green', opacity:0.5}
    };
      
    var trace2 = {
        x: [(json_obj.Neutral/total)*100],
        y: [''],
        name: 'Neutral',
        orientation: 'h',
        type: 'bar',
        marker:{color: 'gray', opacity:0.5}
    };

    var trace3 = {
        x: [(json_obj.Negative/total)*100],
        y: [''],
        name: 'Negative',
        orientation: 'h',
        type: 'bar',
        marker:{color: 'red', opacity:0.5}
    };
      
    var data = [trace1, trace2, trace3];
      
    var layout = {
        title: "<b>Twitter Posts: Sentiment Distribution</b>",
        barmode: 'stack',
        xaxis: {title: "Percent"},
        yaxis: {title: "Sentiment"}
    };
      
    Plotly.newPlot('barChart01', data, layout);
}

// Inserts the html code to show the sample tweets in the classical tweet format.
function insert_tweets(neg_data, neut_data, pos_data){
    // Grab handle to containers where the tweets will go.
    var neg_container = d3.select("#negTweet");
    var neut_container = d3.select("#neutTweet");
    var pos_container = d3.select("#posTweet");

    // Clear the html inside the containers.
    neg_container.html("");
    neut_container.html("");
    pos_container.html("");

    // Shuffle the array to make the posts that appear random.
    shuffled_neg = d3.shuffle(neg_data);
    shuffled_neut = d3.shuffle(neut_data);
    shuffled_pos = d3.shuffle(pos_data);

    // Get 10  posts of each type of sentiment and add them to their respective containers.
    for (let i = 0; i < 10; i++) {
        neg_container.insert("div").html(shuffled_neg[i].Embed_Code);
        neut_container.insert("div").html(shuffled_neut[i].Embed_Code);
        pos_container.insert("div").html(shuffled_pos[i].Embed_Code);
    }

}

// helper function for the collapsible navigation bar.
function openNav(){
    var sideBar = d3.select("#mySidebar");
    sideBar.style("fontSize", "40px");
    sideBar.style("paddingTop", "10%");
    sideBar.style("display", "block");
}

// helper function for the collapsible navigation bar.
function closeNav(){
    var sideBar = d3.select("#mySidebar");
    sideBar.style("display", "none");
}

// Check what file is currently being serviced.
var fileName = document.documentURI.split("/").slice(-1);

// If the home page is being serviced run the init function.
if(fileName == "index.html" || fileName==""){
    init();
}

// Add Event Listeners.
d3.selectAll(".fa-bars").on("click", openNav);
d3.selectAll("#closeNav").on("click", closeNav);