
// Default response
variables.aiSummary = "AI-service unavailable";

try {
    // Perform the HTTP request with a defined timeout (30 seconds)
    cfhttp(
        method = "POST",
        url = "neuroimage.usc.edu", // USC resources typically use HTTPS
        result = "local.res",
        timeout = 30 
    );

    // Verify a successful 200 OK response and that content exists
    if (local.res.statusCode == "200 OK" && isJSON(local.res.fileContent)) {
        local.resultData = deserializeJSON(local.res.fileContent);
        
        // Use Elvis operator or check if key exists to avoid "Key not found" errors
        variables.aiSummary = local.resultData.summary ?: "No summary provided";
    }
} catch (any e) {
    // Log the error for debugging (using your application's logging method)
    writeLog(type="Error", text="USC Neuroimage API Failure: #e.message#");
}
