
// MainFrm.h : CMainFrame 类的接口
//

#pragma once

class CMainFrame : public CFrameWndEx
{
	
protected: // 仅从序列化创建
	CMainFrame();
	DECLARE_DYNCREATE(CMainFrame)

// 特性
public:

// 操作
public:
//	void ChangeSize(CWnd *pWnd, CRect m_rect, int cx, int cy);
// 重写
public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	//virtual BOOL LoadFrame(UINT nIDResource, DWORD dwDefaultStyle = WS_OVERLAPPEDWINDOW | FWS_ADDTOTITLE, CWnd* pParentWnd = NULL, CCreateContext* pContext = NULL);

// 实现
public:
	virtual ~CMainFrame();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:  // 控件条嵌入成员
	CMFCMenuBar       m_wndMenuBar;
	CMFCToolBar       m_wndToolBar;
	CMFCStatusBar     m_wndStatusBar;
	//CMFCToolBarImages m_UserImages;

// 生成的消息映射函数
protected:
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
//	afx_msg void OnViewCustomize();
//	afx_msg LRESULT OnToolbarCreateNew(WPARAM wp, LPARAM lp);
//	afx_msg void OnApplicationLook(UINT id);
//	afx_msg void OnUpdateApplicationLook(CCmdUI* pCmdUI);
	DECLARE_MESSAGE_MAP()

public:
	afx_msg void OnClose();
	//afx_msg void OnIdcancel();
//	afx_msg void OnSize(UINT nType, int cx, int cy);
//	afx_msg void OnBnClickedStart();
//	afx_msg void OnTimer(UINT_PTR nIDEvent);
	HICON m_hIcon;
	afx_msg void OnExit();
};


