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
    res.json(await findAll(req.params.colName, req.query));
})

// route to determine what html file is being served up
app.get("/:fileName", (req, res) =>{
    if(req.params.fileName.includes(".html")){
        res.sendFile(path.join(__dirname, "public/views/", req.params.fileName));
    } 
    else{
        res.sendFile(path.join(__dirname, "public/views/404.html"));
    }
})

// connect to the database
async function main(){
    try{
        await client.connect();
    } catch(e){
        console.error(e);
    }
}

// helper function to find all documents in a collection matching a query.
async function findAll(colName, query={}){
    const cursor = db.collection(colName).find(query);
    const results = await cursor.toArray();

    if(Object.keys(query).length === 0){
        return JSON.stringify(results[0]);
    }
    else{
        return JSON.stringify(results);
    }
}

// Initiate the database connection and begin listening for requests.
main().catch(console.error);
var listener = app.listen(port, ()=>{
    console.log('Your app is listenting on port ' + listener.address().port);
})