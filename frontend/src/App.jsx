import { useState, useRef } from 'react'
import Latex from 'react-latex-next'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isExtracting, setIsExtracting] = useState(false)
  const [ocrResult, setOcrResult] = useState(null)
  const [elapsedMs, setElapsedMs] = useState(null)
  const [error, setError] = useState(null)
  const [viewMode, setViewMode] = useState('latex') // 'latex' | 'code'
  const fileInputRef = useRef(null)

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://164.52.192.138:8080/v1'

  const handleFileSelect = (selectedFile) => {
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile)
      setError(null)
      setOcrResult(null)
      setElapsedMs(null)
    } else {
      setError('Please select a PDF file')
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const droppedFile = e.dataTransfer.files[0]
    handleFileSelect(droppedFile)
  }

  const handleFileInputChange = (e) => {
    const selectedFile = e.target.files[0]
    handleFileSelect(selectedFile)
  }

  const handleExtract = async () => {
    if (!file) {
      setError('Please select a file first')
      return
    }

    setIsExtracting(true)
    setError(null)
    setElapsedMs(null)

    try {
      const startTime = performance.now()
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_BASE_URL}/invoice/ocr-text`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setOcrResult(data)
      const endTime = performance.now()
      setElapsedMs(endTime - startTime)
    } catch (err) {
      setError(err.message || 'Failed to extract OCR text. Please try again.')
      setOcrResult(null)
    } finally {
      setIsExtracting(false)
    }
  }

  const handleClear = () => {
    setFile(null)
    setOcrResult(null)
    setElapsedMs(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="app-container">
      <div className="app-content">
        <h1>Invoice OCR Extractor</h1>
        <p className="subtitle">Upload a PDF invoice to extract LaTeX text</p>

        {/* File Upload Area */}
        <div
          className={`upload-area ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            onChange={handleFileInputChange}
            style={{ display: 'none' }}
          />
          <div className="upload-content">
            {file ? (
              <>
                <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <p className="file-name">{file.name}</p>
                <p className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </>
            ) : (
              <>
                <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <p className="upload-text">Drag and drop your PDF here</p>
                <p className="upload-subtext">or click to browse</p>
              </>
            )}
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="error-message">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            {error}
          </div>
        )}

        {/* Action Buttons */}
        <div className="button-group">
          <button
            className="extract-button"
            onClick={handleExtract}
            disabled={!file || isExtracting}
          >
            {isExtracting ? (
              <>
                <svg className="spinner" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" strokeDasharray="32" strokeDashoffset="32">
                    <animate attributeName="stroke-dasharray" dur="2s" values="0 32;16 16;0 32;0 32" repeatCount="indefinite" />
                    <animate attributeName="stroke-dashoffset" dur="2s" values="0;-16;-32;-32" repeatCount="indefinite" />
                  </circle>
                </svg>
                Extracting...
              </>
            ) : (
              <>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                  <polyline points="10 9 9 9 8 9" />
                </svg>
                Extract OCR
              </>
            )}
          </button>

          {ocrResult && (
            <button className="clear-button" onClick={handleClear}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
              </svg>
              Clear
            </button>
          )}
        </div>

        {/* OCR Result Display */}
        {ocrResult && (
          <div className="result-container">
            <div className="result-header">
              <h2>Extracted LaTeX</h2>
              <div className="result-info">
                <span>{ocrResult.total_pages} page{ocrResult.total_pages !== 1 ? 's' : ''}</span>
                {elapsedMs !== null && (
                  <span>Time taken: {(elapsedMs / 1000).toFixed(2)} s</span>
                )}
              </div>
            </div>
            {/* Table: always shown */}
            <div className="table-section">
              {ocrResult.table?.columns?.length > 0 ? (
                <div className="table-wrapper">
                  <table className="result-table">
                    <thead>
                      <tr>
                        {ocrResult.table.columns.map((col, i) => (
                          <th key={i}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {ocrResult.table.data.map((row, i) => (
                        <tr key={i}>
                          {ocrResult.table.columns.map((col, j) => (
                            <td key={j}>{row[col] ?? ''}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <details className="json-details">
                    <summary>View as JSON</summary>
                    <pre className="latex-output json-preview">
                      {JSON.stringify(ocrResult.table.data, null, 2)}
                    </pre>
                  </details>
                </div>
              ) : (
                <p className="no-table">No table found.</p>
              )}
            </div>

            <div className="view-toggle">
              <button
                type="button"
                className={`toggle-button ${viewMode === 'latex' ? 'active' : ''}`}
                onClick={() => setViewMode('latex')}
              >
                LaTeX
              </button>
              <button
                type="button"
                className={`toggle-button ${viewMode === 'code' ? 'active' : ''}`}
                onClick={() => setViewMode('code')}
              >
                Code
              </button>
            </div>
            <div className="result-content">
              {viewMode === 'latex' && (
                <div className="latex-render">
                  <Latex>{ocrResult.full_text}</Latex>
                </div>
              )}
              {viewMode === 'code' && (
                <pre className="latex-output">{ocrResult.full_text}</pre>
              )}
            </div>
            {ocrResult.pages && ocrResult.pages.length > 1 && (
              <div className="pages-section">
                <h3>Page-by-Page Results</h3>
                {ocrResult.pages.map((page, index) => (
                  <div key={index} className="page-result">
                    <h4>Page {page.page}</h4>
                    {viewMode === 'code' ? (
                      <pre className="latex-output">{page.latex}</pre>
                    ) : (
                      <div className="latex-render">
                        <Latex>{page.latex}</Latex>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
