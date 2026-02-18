<cfsetting enablecfoutputonly="true">

<cfparam name="url.task" default="analyze">
<cfparam name="form.data" default="{}">

<cfset tempfile_in = GetTempFile(GetTempDirectory(), "brain_")>
<cffile action="write" file="#tempfile_in#" output="#form.data#" charset="utf-8">

<cftry>
    <cfexecute name="python"
               arguments="/app/ai_core.py #tempfile_in#"
               timeout="60"
               variable="result">
    </cfexecute>
    <cfoutput>#result#</cfoutput>
<cfcatch type="any">
    <cfoutput>Error: #cfcatch.message#</cfoutput>
</cfcatch>
</cftry>

<cffile action="delete" file="#tempfile_in#">
<cfsetting enablecfoutputonly="false">
