const baseURL = "http://localhost:8080/"

function init(){

    let chartData = null;


    fetch(baseURL+'results/current_pricing')
    .then(response=> response.json())
    .then(data => {
        buildTable(data);
    });

    fetch(baseURL+'results/gas_history')
    .then(response=> response.json())
    .then(data => {
        buildLineChart(data);
    });

    buildSentimentChart(0);

}


function buildTable(data) {
    // First, clear out any existing data
    var tbody = d3.select("tbody");
    tbody.html("");

    // Parse data into JSON format
    const myJSON = JSON.parse(data)

    //Add row header
    let row = tbody.append("tr");
    let cell = row.append("td");
    cell.text("Current Average");

    //Iterate through data and add todays gas prices
    for (const x in myJSON.todays_prices){
        let cell = row.append("td");
        cell.text(myJSON.todays_prices[x]);
    }

    //Add row header
    row = tbody.append("tr");
    cell = row.append("td");
    cell.text("Predicted Average");

    for (const x in myJSON.todays_prices){
        let cell = row.append("td");
        cell.text(myJSON.todays_prices[x]);
    }
}

function jsonValueExtract(sample, colName){
    let results = [];
    for( const x in sample[colName]){
        results.push(sample[colName][x]);
    }
    return results;
}

function buildLineChart(sample) {
    // Parse data into JSON format
    const myJSON = JSON.parse(sample)

    const gasDates = jsonValueExtract(myJSON, "Date");

    var trace1={
        x:gasDates,
        y:jsonValueExtract(myJSON, "Weekly US Regular Price"),
        type: 'scatter',
        name: 'Regular'
    };

    var trace2={
        x:gasDates,
        y:jsonValueExtract(myJSON, "Weekly US Midgrade Price"),
        type: 'scatter',
        name: 'Midgrade'
    };

    var trace3={
        x:gasDates,
        y:jsonValueExtract(myJSON, "Weekly US Premium Price"),
        type: 'scatter',
        name: 'Premium'
    };

    var trace4={
        x:gasDates,
        y:jsonValueExtract(myJSON, "Weekly US No2 Diesel Price"),
        type: 'scatter',
        name: 'Diesel'
    };
    
    var lineLayout={
        title: "<b>Weekly Avg. Gas Price by Fuel Type</b>",
        xaxis: {title: "Date"},
        yaxis: {title: "Price (USD)"}
    }

    var data = [trace1, trace2, trace3, trace4];

    Plotly.newPlot('lineChart01', data, lineLayout);

}

function buildSentimentChart(data){

    var json_obj = {"Positive": 40, "Neutral": 10, "Negative": 50}

    var trace1 = {
        x: [json_obj.Positive],
        y: [''],
        name: 'Positive',
        orientation: 'h',
        type: 'bar',
        marker:{color: 'green', opacity:0.5}
    };
      
    var trace2 = {
        x: [json_obj.Neutral],
        y: [''],
        name: 'Neutral',
        orientation: 'h',
        type: 'bar',
        marker:{color: 'gray', opacity:0.5}
    };

    var trace3 = {
        x: [json_obj.Negative],
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

function openNav(){
    var sideBar = d3.select("#mySidebar");
    sideBar.style("fontSize", "40px");
    sideBar.style("paddingTop", "10%");
    sideBar.style("display", "block");
}

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