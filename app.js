// Import Statements
const express = require('express');
const path = require('path')
const {MongoClient} = require('mongodb');
const exp = require('constants');

// Set up express
const app = express();
app.use(require("cors")())
app.use(require('body-parser').json())
app.use(express.static(path.join(__dirname, '/public')));

// Define constants
const uri = "mongodb://127.0.0.1:27017/gas_app"
const dbName = "gas_app"
const port = 8080;

// Set up database connection
const client = new MongoClient(uri);
const db = client.db(dbName);



// base route
app.get("/", (req, res) =>{
    res.sendFile(path.join(__dirname, "public/views/index.html"));
})

// route to retrieve whole collection
app.get("/results/:colName", async (req, res) =>{
    res.json(await findAll(req.params.colName));
})

app.get("/:fileName", (req, res) =>{
    if(req.params.fileName.includes(".html")){
        res.sendFile(path.join(__dirname, "public/views/", req.params.fileName));
    } 
    else{
        res.sendFile(path.join(__dirname, "public/views/404.html"));
    }
})

async function main(){
    try{
        await client.connect();
    } catch(e){
        console.error(e);
    }
}

async function findAll(colName){
    const cursor = db.collection(colName).find({});
    const results = await cursor.toArray();

    return JSON.stringify(results[0]);
}


async function init(){
    // Set up database connection
    const client = new MongoClient(uri);
    const db = client.db(dbName);

    // Initialize containers for database data retrieval
    let tableData = null;
    let chartData = null;
    let sentimentData = null;
    let twitterData = null;


    try{
        await client.connect();
        chartData = await findAll(db, "gas_history");
        tableData = await findAll(db, "current_pricing");
    } catch(e){
        console.error(e);
    }finally{
        await client.close();
    }

    buildTable(tableData);
}





//init();
main().catch(console.error);
var listener = app.listen(port, ()=>{
    console.log('Your app is listenting on port ' + listener.address().port);
})