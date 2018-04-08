// AxisState.cpp : 实现文件
//

#include "stdafx.h"
#include "CV_System.h"
#include "AxisState.h"
#include "afxdialogex.h"


// CAxisState 对话框

IMPLEMENT_DYNAMIC(CAxisState, CDialog)

CAxisState::CAxisState(CWnd* pParent /*=NULL*/)
	: CDialog(CAxisState::IDD, pParent)
	, m_axisnum(_T(""))
{

}

CAxisState::~CAxisState()
{
}

void CAxisState::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_CBString(pDX, IDC_COMBO1, m_axisnum);
}


BEGIN_MESSAGE_MAP(CAxisState, CDialog)
END_MESSAGE_MAP()


// CAxisState 消息处理程序


void CAxisState::PostNcDestroy()
{
	// TODO: 在此添加专用代码和/或调用基类

	CDialog::PostNcDestroy();
	delete this;
}


void CAxisState::OnCancel()
{
	// TODO: 在此添加专用代码和/或调用基类

	//CDialog::OnCancel();
	DestroyWindow();
}


BOOL CAxisState::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO:  在此添加额外的初始化
	((CComboBox*)GetDlgItem(IDC_COMBO1))->AddString(_T("1"));
	((CComboBox*)GetDlgItem(IDC_COMBO1))->AddString(_T("2"));
	((CComboBox*)GetDlgItem(IDC_COMBO1))->AddString(_T("3"));
	((CComboBox*)GetDlgItem(IDC_COMBO1))->AddString(_T("4"));

	((CComboBox*)GetDlgItem(IDC_COMBO1))->SetCurSel(0);
	return TRUE;  // return TRUE unless you set the focus to a control
	// 异常: OCX 属性页应返回 FALSE
}
