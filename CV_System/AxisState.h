#pragma once


// CAxisState 对话框

class CAxisState : public CDialog
{
	DECLARE_DYNAMIC(CAxisState)

public:
	CAxisState(CWnd* pParent = NULL);   // 标准构造函数
	virtual ~CAxisState();

// 对话框数据
	enum { IDD = IDD_AxisState };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
	virtual void PostNcDestroy();
	virtual void OnCancel();
public:
	virtual BOOL OnInitDialog();
	CString m_axisnum;
};
