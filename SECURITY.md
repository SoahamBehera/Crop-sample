# 🔒 Security Policy

## Overview

CultivaSense is committed to ensuring the security and privacy of our users' data. This document outlines our security practices, vulnerability reporting procedures, and guidelines for maintaining a secure application environment.

## 🛡️ Supported Versions

We actively support and provide security updates for the following versions:

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 1.x.x   | ✅ Yes            | Active Development |
| < 1.0   | ❌ No             | End of Life |

**Recommendation:** Always use the latest stable release to ensure you have the most recent security patches and improvements.

---

## 🚨 Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### **Reporting Process**

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. **Email us directly** at: **dasouvik122005@gmail.com**
3. **Include the following information:**
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Suggested fix (if available)
   - Your contact information for follow-up

### **Response Timeline**

- **Initial Response:** Within 48 hours of report submission
- **Status Update:** Within 7 days with assessment and action plan
- **Resolution:** Critical vulnerabilities will be addressed within 30 days
- **Disclosure:** Coordinated disclosure after patch is released

### **Recognition**

We appreciate security researchers who help us maintain a secure platform. With your permission, we will:
- Acknowledge your contribution in our security advisories
- Credit you in our release notes (unless you prefer to remain anonymous)

---

## 🔐 Security Best Practices

### **For Developers**

#### 1. **Environment Configuration**

```bash
# Never commit sensitive credentials to version control
# Always use environment variables for sensitive data

# Copy the example environment file
cp .env.example .env

# Update with your secure values
SECRET_KEY=<generate-strong-random-key>
FLASK_DEBUG=False  # NEVER enable debug mode in production
```

**Generate a secure SECRET_KEY:**
```python
import secrets
print(secrets.token_hex(32))
```

#### 2. **Dependency Management**

- **Keep dependencies updated:** Regularly update packages to patch known vulnerabilities
  ```bash
  pip list --outdated
  pip install --upgrade -r requirements.txt
  ```

- **Use virtual environments:** Isolate project dependencies
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\Scripts\activate
  ```

- **Audit dependencies:** Check for known security vulnerabilities
  ```bash
  pip install safety
  safety check
  ```

#### 3. **File Upload Security**

The application implements several security measures for file uploads:

- **File size limits:** Maximum 5MB per upload
- **File type validation:** Only JPG and PNG images accepted
- **Secure filename handling:** Sanitized to prevent directory traversal
- **Temporary storage:** Uploaded files are stored in isolated `uploads/` directory
- **File cleanup:** Regular cleanup of temporary files recommended

**Additional Recommendations:**
```python
# Implement in production:
- Content-Type verification
- Virus scanning for uploaded files
- Rate limiting on upload endpoints
- Separate storage domain for user uploads
```

#### 4. **Input Validation**

All user inputs are validated:
- **Soil parameters:** Range validation (N: 0-140, P: 5-145, K: 5-205, pH: 3.5-9.5)
- **Temperature:** -10°C to 50°C
- **Humidity:** 0-100%
- **Rainfall:** 0-300mm
- **SQL Injection Prevention:** Using parameterized queries and ORM
- **XSS Prevention:** Input sanitization and output encoding

#### 5. **Authentication & Authorization** (Future Implementation)

Currently, CultivaSense is a public tool without user authentication. For production deployments with user accounts:

**Planned Security Features:**
- [ ] Secure password hashing (bcrypt/Argon2)
- [ ] Session management with secure cookies
- [ ] CSRF protection tokens
- [ ] Rate limiting on authentication endpoints
- [ ] Multi-factor authentication (MFA)
- [ ] Role-based access control (RBAC)

---

### **For Deployment**

#### 1. **Production Configuration**

```bash
# Production Environment Variables
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=<strong-random-secret-key>

# Use production-grade WSGI server
# DO NOT use Flask development server in production
```

**Recommended Production Setup:**
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

#### 2. **HTTPS/TLS Configuration**

**Always use HTTPS in production:**
- Obtain SSL/TLS certificates (Let's Encrypt recommended)
- Configure reverse proxy (Nginx/Apache) with SSL
- Enforce HTTPS redirects
- Enable HSTS (HTTP Strict Transport Security)

**Example Nginx Configuration:**
```nginx
server {
    listen 443 ssl http2;
    server_name cultivasense.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### 3. **Database Security** (If Applicable)

For future database implementations:
- Use parameterized queries to prevent SQL injection
- Encrypt sensitive data at rest
- Implement database access controls
- Regular database backups with encryption
- Separate database credentials per environment

#### 4. **API Security**

**Current Implementation:**
- Input validation on all endpoints
- Error handling without information disclosure
- CORS configuration for cross-origin requests

**Recommended Enhancements:**
- API rate limiting (e.g., Flask-Limiter)
- API key authentication for programmatic access
- Request/response logging for audit trails
- API versioning for backward compatibility

#### 5. **Monitoring & Logging**

**Security Logging:**
```python
# Implement comprehensive logging
- Failed authentication attempts
- Unusual file upload patterns
- API rate limit violations
- Application errors and exceptions
- Security-relevant events
```

**Log Security:**
- Do NOT log sensitive data (passwords, API keys, PII)
- Implement log rotation and retention policies
- Secure log storage with restricted access
- Regular log analysis for security incidents

---

## 🔍 Security Features

### **Current Implementation**

✅ **Input Validation:** All user inputs are validated and sanitized  
✅ **File Upload Security:** Size limits, type validation, secure filename handling  
✅ **Error Handling:** Generic error messages to prevent information disclosure  
✅ **Dependency Management:** Regular updates to address known vulnerabilities  
✅ **Environment Variables:** Sensitive configuration via environment variables  
✅ **CORS Configuration:** Controlled cross-origin resource sharing  

### **Planned Enhancements**

🔄 **Rate Limiting:** Prevent abuse and DDoS attacks  
🔄 **Content Security Policy (CSP):** Mitigate XSS attacks  
🔄 **Security Headers:** Comprehensive HTTP security headers  
🔄 **API Authentication:** Secure API access with tokens  
🔄 **Audit Logging:** Comprehensive security event logging  
🔄 **Automated Security Scanning:** CI/CD integration for vulnerability detection  

---

## 🛠️ Security Checklist for Deployment

### **Pre-Deployment**

- [ ] All dependencies updated to latest secure versions
- [ ] `FLASK_DEBUG=False` in production environment
- [ ] Strong `SECRET_KEY` generated and configured
- [ ] Environment variables properly configured (no hardcoded secrets)
- [ ] File upload directory has restricted permissions
- [ ] Production WSGI server configured (Gunicorn/uWSGI)
- [ ] SSL/TLS certificates obtained and configured
- [ ] Security headers configured in reverse proxy
- [ ] Rate limiting implemented on critical endpoints
- [ ] Error handling configured to prevent information disclosure
- [ ] Logging configured with appropriate security events
- [ ] Backup and disaster recovery plan in place

### **Post-Deployment**

- [ ] Security monitoring and alerting configured
- [ ] Regular security audits scheduled
- [ ] Incident response plan documented
- [ ] Security patch management process established
- [ ] Regular dependency vulnerability scans
- [ ] Log analysis and review procedures
- [ ] Penetration testing conducted (if applicable)

---

## 📋 Common Vulnerabilities & Mitigations

### **1. Cross-Site Scripting (XSS)**

**Risk:** Malicious scripts injected into web pages  
**Mitigation:**
- Input sanitization and validation
- Output encoding for user-generated content
- Content Security Policy (CSP) headers
- Use of templating engines with auto-escaping (Jinja2)

### **2. SQL Injection**

**Risk:** Malicious SQL queries via user input  
**Mitigation:**
- Parameterized queries and prepared statements
- ORM usage (SQLAlchemy recommended)
- Input validation and sanitization
- Least privilege database access

### **3. Cross-Site Request Forgery (CSRF)**

**Risk:** Unauthorized actions on behalf of authenticated users  
**Mitigation:**
- CSRF tokens on all state-changing operations
- SameSite cookie attribute
- Verify Origin/Referer headers
- Use Flask-WTF for form protection

### **4. Insecure Direct Object References (IDOR)**

**Risk:** Unauthorized access to resources  
**Mitigation:**
- Implement proper authorization checks
- Use indirect references (UUIDs instead of sequential IDs)
- Validate user permissions on every request

### **5. Security Misconfiguration**

**Risk:** Default configurations exposing vulnerabilities  
**Mitigation:**
- Disable debug mode in production
- Remove unnecessary features and endpoints
- Keep all software updated
- Implement security headers
- Regular security audits

### **6. Sensitive Data Exposure**

**Risk:** Unprotected sensitive information  
**Mitigation:**
- Encrypt data in transit (HTTPS/TLS)
- Encrypt sensitive data at rest
- Secure credential storage (environment variables)
- Implement proper access controls
- No sensitive data in logs or error messages

---

## 🔄 Security Update Policy

### **Update Frequency**

- **Critical vulnerabilities:** Immediate patch within 24-48 hours
- **High severity:** Patch within 7 days
- **Medium severity:** Patch within 30 days
- **Low severity:** Included in next regular release

### **Notification Channels**

Security updates will be announced via:
- GitHub Security Advisories
- Release notes on GitHub
- Email to registered users (future implementation)
- Project README.md updates

---

## 📞 Contact Information

### **Security Team**

- **Email:** cultivasense_test@gmail.com
- **GitHub:** [@SoahamBehera](https://github.com/SoahamBehera)
- **Response Time:** Within 48 hours for security issues

### **General Support**

- **Website:** www.cultivasense.com
- **GitHub Issues:** For non-security bugs and feature requests
- **Documentation:** See [README.md](README.md)

---

## 📚 Additional Resources

### **Security Guidelines**

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Flask Security Best Practices](https://flask.palletsprojects.com/en/latest/security/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [CWE Top 25 Most Dangerous Software Weaknesses](https://cwe.mitre.org/top25/)

### **Tools for Security Testing**

- **OWASP ZAP:** Web application security scanner
- **Bandit:** Python security linter
- **Safety:** Python dependency vulnerability checker
- **npm audit:** Node.js dependency security audit (if using Node.js tools)

---

## 📜 Compliance & Standards

CultivaSense aims to align with industry-standard security practices:

- **OWASP Application Security Verification Standard (ASVS)**
- **CWE/SANS Top 25 Most Dangerous Software Errors**
- **ISO/IEC 27001** (Information Security Management)
- **GDPR** (General Data Protection Regulation) - for future user data handling

---

## 🔐 Responsible Disclosure

We believe in responsible disclosure and will work with security researchers to:

1. **Acknowledge** receipt of vulnerability reports promptly
2. **Investigate** and validate reported vulnerabilities
3. **Develop** and test patches in a timely manner
4. **Coordinate** disclosure timing with the reporter
5. **Credit** researchers appropriately (with permission)

**We commit to:**
- Not pursuing legal action against researchers who follow responsible disclosure
- Keeping reporters informed throughout the remediation process
- Publicly acknowledging contributions (unless anonymity is requested)

---

## 📝 Version History

| Version | Date       | Changes |
|---------|------------|---------|
| 1.0.0   | 2026-01-29 | Initial security policy release |

---

## ⚖️ License

This security policy is part of the CultivaSense project and is subject to the same license terms.

---

<div align="center">

**🔒 Security is a shared responsibility**

*If you see something, say something. Together, we can keep CultivaSense secure.*

**© 2026 CultivaSense Security Team**

</div>
