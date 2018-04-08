#pragma once


// CShowDlg 对话框
#include "CvvImage.h"
class CShowDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CShowDlg)

public:
	CShowDlg(CWnd* pParent = NULL);   // 标准构造函数
	virtual ~CShowDlg();

// 对话框数据
	enum { IDD = IDD_CV_SYSTEM_FORM };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()

private:       
	CRect rect;
	CStatic* pStc; //标识图像显示的Picture控件
	CDC* pDC; //视频显示控件设备上下文
	HDC hDC; //视频显示控件设备句柄
	CvCapture* capture; //视频获取结构
public:
	virtual BOOL OnInitDialog();
	afx_msg void OnBnClickedStart();
};
