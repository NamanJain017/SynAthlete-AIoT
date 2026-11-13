# Fill the table of contents, the lists of figures and tables and the page numbers of
# SynAthlete_Report.docx with Microsoft Word, save it, and export a PDF next to it.
# Run after `python -m src.paper.report_docx`:  powershell -File report\update_fields.ps1
# Progress is written to report\update_fields.log.
$ErrorActionPreference = "Stop"
$docx = Join-Path $PSScriptRoot "SynAthlete_Report.docx"
$pdf = Join-Path $PSScriptRoot "SynAthlete_Report.pdf"
$log = Join-Path $PSScriptRoot "update_fields.log"
function Step($msg) { "$(Get-Date -Format HH:mm:ss) $msg" | Add-Content $log }
Set-Content $log ""
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$word.Options.CheckGrammarAsYouType = $false
$word.Options.CheckSpellingAsYouType = $false
try {
    Step "opening"
    $doc = $word.Documents.Open($docx, $false, $false, $false)
    Step "opened; updating $($doc.TablesOfContents.Count) contents lists"
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    Step "lists updated; updating page fields"
    foreach ($s in $doc.Sections) { foreach ($h in $s.Headers) { $h.Range.Fields.Update() | Out-Null } }
    foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
    Step ("pages: " + $doc.ComputeStatistics(2))
    Step "saving"
    $doc.Save()
    Step "exporting PDF"
    $doc.SaveAs2($pdf, 17)
    Step "PDF written"
    $doc.Close($false)
}
catch { Step ("ERROR: " + $_.Exception.Message) }
finally {
    $word.Quit()
    Step "done"
}
